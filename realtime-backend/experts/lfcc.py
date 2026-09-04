from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from config import HUB_EXPERTS, TARGET_SAMPLE_RATE, WINDOW_SEC
from .hub import ensure_checkpoint, torch_load_checkpoint
from .protocol import Score

logger = logging.getLogger("realtime_backend.experts")

_WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SAMPLE_RATE)


class LFCCLCNNExpert:
    """Adapter for LFCC + LCNN experts.

    The checkpoint is fetched from the Hub and loaded into the vendored
    architecture in :mod:`experts.lfcc_model`, which is a copy of the trainer's
    ``LFCCLCNNWithFeatureExtraction``. Higher logit = more spoof evidence. The
    128-dim penultimate embedding is returned for fusion.

    A single adapter class serves every LFCC-LCNN checkpoint because they share
    one architecture (n_lfcc=20, deltas, 128-dim embedding). ``hub_key`` selects
    which entry in ``config.HUB_EXPERTS`` to download, and ``name`` is the label
    the expert reports to fusion / the response contract. The default is the
    shipped ``hybrid`` checkpoint (6 languages, 130 spoof generators).

    BAND GATE. Some checkpoints are trained with a parity band gate -- every
    window lowpassed at ``band_gate_hz`` -- because the training corpus made the
    near-Nyquist band a label proxy (training resampled with librosa/soxr, which
    brickwalls 7.9-8 kHz; this backend resamples with scipy, which does not, so
    the band was a train/serve mismatch that made in-training generators read
    bonafide live). The cutoff is read from the checkpoint, NOT hardcoded here,
    and applied inside :meth:`score`. A gated checkpoint served ungated is
    biased spoofward, so this must never be skipped or duplicated by callers.
    """

    name = "hybrid"
    model_version = "lfcc-unwired"

    def __init__(
        self,
        cache_dir: Path,
        device: str = "cpu",
        hub_key: str = "hybrid",
        name: str | None = None,
    ):
        self.name = name or hub_key
        spec = HUB_EXPERTS[hub_key]
        self.device = torch.device(device)
        if spec.get("local_only"):
            # Local-only checkpoint (e.g. a fine-tune not published to the Hub):
            # load straight from the cache dir and never touch the network. This
            # deliberately skips ensure_checkpoint so its HEAD/size-diff staleness
            # check can never re-download or clobber a file that has no Hub twin.
            dest = Path(cache_dir) / spec["local_name"]
            if not (dest.exists() and dest.stat().st_size > 0):
                raise RuntimeError(
                    f"Local-only checkpoint for expert {self.name!r} not found at {dest}.\n"
                    f"Place {spec['local_name']!r} in {cache_dir} (it is not on the Hub)."
                )
            self.checkpoint_path = dest
        else:
            self.checkpoint_path = ensure_checkpoint(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                cache_dir=cache_dir,
                local_name=spec["local_name"],
            )
        self._blob = torch_load_checkpoint(self.checkpoint_path, map_location=str(self.device))
        self.model = None
        self._wire_model()

    def _wire_model(self) -> None:
        from .lfcc_model.lfcc_lcnn import LFCCLCNNWithFeatureExtraction

        # Architecture params are fixed by the published checkpoint: n_lfcc=20
        # with deltas -> 60 features, 128-dim embedding. dropout is irrelevant
        # at inference (eval mode) and does not affect state_dict keys.
        model = LFCCLCNNWithFeatureExtraction(
            sample_rate=TARGET_SAMPLE_RATE,
            n_lfcc=20,
            with_deltas=True,
            embedding_dim=128,
            dropout=0.0,
        )
        if isinstance(self._blob, dict):
            state = self._blob.get("model_state_dict", self._blob)
        else:
            state = self._blob
        # strict=True: fail loudly at startup on any architecture mismatch
        # rather than silently scoring with a half-loaded model.
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        self.model = model
        self.model_version = Path(self.checkpoint_path).stem

        best_eer = self._blob.get("best_eer") if isinstance(self._blob, dict) else None

        # Read the gate from the checkpoint that was trained with it. Hardcoding
        # it here would let the two drift apart silently.
        gate = self._blob.get("band_gate_hz") if isinstance(self._blob, dict) else None
        self.band_gate_hz = float(gate) if gate else None
        if self.band_gate_hz:
            nyq = TARGET_SAMPLE_RATE / 2
            if not (0 < self.band_gate_hz < nyq):
                raise RuntimeError(
                    f"Expert {self.name!r} declares band_gate_hz={self.band_gate_hz}, "
                    f"which is not inside (0, {nyq}). Refusing to load rather than "
                    f"score with a nonsensical gate."
                )
            # Precompute the rFFT bin mask once; score() is on the hot path.
            freqs = np.fft.rfftfreq(_WINDOW_SAMPLES, 1.0 / TARGET_SAMPLE_RATE)
            self._gate_mask = freqs >= self.band_gate_hz
            logger.info(
                "Expert %s uses a %.0f Hz parity band gate (from checkpoint); "
                "windows are lowpassed before scoring",
                self.name,
                self.band_gate_hz,
            )
        else:
            self._gate_mask = None

        logger.info(
            "Loaded expert %s version=%s from %s (best_eer=%s, band_gate=%s)",
            self.name,
            self.model_version,
            self.checkpoint_path,
            f"{best_eer:.4f}" if isinstance(best_eer, (int, float)) else "n/a",
            f"{self.band_gate_hz:.0f} Hz" if self.band_gate_hz else "none",
        )
        self._blob = None  # release checkpoint (optimizer state etc.) after load

    def score(self, audio_window: np.ndarray) -> Score:
        if self.model is None:
            raise RuntimeError("LFCCLCNNExpert is not wired; _wire_model() did not run")
        window = np.asarray(audio_window, dtype=np.float32).reshape(-1)
        # The pipeline emits exactly _WINDOW_SAMPLES; pad/crop defensively so a
        # caller passing a short/long window never crashes the stream.
        if window.size < _WINDOW_SAMPLES:
            window = np.pad(window, (0, _WINDOW_SAMPLES - window.size))
        elif window.size > _WINDOW_SAMPLES:
            window = window[:_WINDOW_SAMPLES]

        # Parity band gate, if this checkpoint was trained with one. Must happen
        # here (after pad/crop, so the precomputed bin mask matches) and for
        # every window, including the padded ones.
        if self._gate_mask is not None:
            spec = np.fft.rfft(window)
            spec[self._gate_mask] = 0.0
            window = np.fft.irfft(spec, n=window.size).astype(np.float32)

        tensor = torch.from_numpy(window).to(self.device).unsqueeze(0)  # (1, N)
        with torch.no_grad():
            logit_t, embedding_t = self.model(tensor, return_embedding=True)
        logit = float(logit_t.squeeze().item())
        embedding = embedding_t.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()
        return {"logit": logit, "embedding": embedding, "model_version": self.model_version}

    def __repr__(self) -> str:  # aids the /health and startup logs
        gate = f"{self.band_gate_hz:.0f}Hz" if getattr(self, "band_gate_hz", None) else "none"
        return f"LFCCLCNNExpert(name={self.name!r}, version={self.model_version!r}, gate={gate})"
