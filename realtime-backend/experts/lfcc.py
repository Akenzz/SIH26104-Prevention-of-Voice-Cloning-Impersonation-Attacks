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
        logger.info(
            "Loaded expert %s version=%s from %s (best_eer=%s)",
            self.name,
            self.model_version,
            self.checkpoint_path,
            f"{best_eer:.4f}" if isinstance(best_eer, (int, float)) else "n/a",
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

        tensor = torch.from_numpy(window).to(self.device).unsqueeze(0)  # (1, N)
        with torch.no_grad():
            logit_t, embedding_t = self.model(tensor, return_embedding=True)
        logit = float(logit_t.squeeze().item())
        embedding = embedding_t.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()
        return {"logit": logit, "embedding": embedding, "model_version": self.model_version}
