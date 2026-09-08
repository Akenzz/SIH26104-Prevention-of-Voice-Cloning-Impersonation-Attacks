"""
experts/ssl_antispoof.py — Adapter for TakHemlata/SSL_Anti-spoofing (AASIST)

fairseq removed: the XLSR-2 backbone (facebook/wav2vec2-xls-r-300m) is now
loaded from HuggingFace via ``transformers``.  This is architecturally identical
to the fairseq xlsr2_300m.pt used in training (same weights, same 1024-dim
output), so the saved AASIST head weights load correctly.

The ``best_SSL_model_LA.pth`` file from Akenzz/SIH-Models contains ONLY the
AASIST graph-attention head weights (not the XLSR backbone), so we load it
with strict=False to silently skip any ssl_model.model.* keys that are absent
from the checkpoint.

Score output (same contract as others):
    {"logit": float, "embedding": list[float], "model_version": str}

Label convention (IMPORTANT — never flip):
    logit > 0  →  spoof / fake  (label 1)
    logit < 0  →  bonafide / real  (label 0)
"""
from __future__ import annotations
import logging
import sys
import os
from pathlib import Path
import numpy as np
import torch
from config import HUB_EXPERTS
from .hub import ensure_checkpoint
from .protocol import Score

logger = logging.getLogger("realtime_backend.experts")

_WINDOW_SAMPLES = 64600
_SSL_REPO = Path(__file__).resolve().parent.parent.parent / "ssl-detector"


def _add_ssl_repo_to_path():
    if str(_SSL_REPO) not in sys.path:
        sys.path.insert(0, str(_SSL_REPO))


class SSLExpert:
    name = "ssl"
    model_version = "ssl-aasist-xlsr300m-hf"

    def __init__(self, cache_dir: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.model = None
        self._last_hidden = None
        self._wire_model()

    def _wire_model(self):
        _add_ssl_repo_to_path()
        cwd_backup = os.getcwd()
        os.chdir(str(_SSL_REPO))

        try:
            import model as ssl_module

            # HF backbone downloads go to model_cache/hf_cache (D drive).
            # This matches HF_HOME set in .env, but passing it explicitly here
            # is belt-and-suspenders in case the env var hasn't propagated yet.
            hf_cache_dir = Path(self.cache_dir) / "hf_cache"
            hf_cache_dir.mkdir(parents=True, exist_ok=True)

            # Build the AASIST model.  SSLModel.__init__ now loads the XLSR
            # backbone from HuggingFace (no fairseq required).
            self.model = ssl_module.Model(None, self.device, cache_dir=str(hf_cache_dir))

            # ── Resolve checkpoint ─────────────────────────────────────────────
            # Primary: model_cache/ (downloaded from Akenzz/SIH-Models).
            spec = HUB_EXPERTS["ssl"]
            ckpt_path = Path(self.cache_dir) / spec["local_name"]

            if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
                # Fallback: legacy path inside ssl-detector/ (for teammates
                # who copied the file there manually before the HF upload).
                legacy = _SSL_REPO / "best_SSL_model_LA.pth"
                if legacy.exists() and legacy.stat().st_size > 0:
                    logger.info("SSL: using legacy checkpoint at %s", legacy)
                    ckpt_path = legacy
                else:
                    logger.info("SSL: checkpoint not found locally — downloading from HF.")
                    ckpt_path = ensure_checkpoint(
                        repo_id=spec["repo_id"],
                        filename=spec["filename"],
                        cache_dir=self.cache_dir,
                        local_name=spec["local_name"],
                    )

            # ── Load weights ───────────────────────────────────────────────────
            # The .pth file contains AASIST head weights AND the fine-tuned XLSR
            # backbone (saved under ssl_model.model.* with fairseq key names).
            #
            # Step 1: remap backbone keys (fairseq → HF) into the HF Wav2Vec2Model.
            # Step 2: load full state dict with strict=False for the AASIST head;
            #         missing ssl_model.model.* keys are expected (already loaded).
            blob = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            # Some checkpoints wrap state_dict in a dict; others are bare.
            state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob

            # Step 1 — fine-tuned backbone weights
            self.model.ssl_model.load_finetuned_backbone(state)

            # Step 2 — AASIST head weights (ssl_model.model.* keys will be missing
            # because they are not in the HF Wav2Vec2Model's state dict; that is fine)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            backbone_missing = [k for k in missing if k.startswith("ssl_model.model.")]
            other_missing    = [k for k in missing if not k.startswith("ssl_model.model.")]
            if backbone_missing:
                logger.debug(
                    "SSL: %d XLSR backbone keys loaded via key-remap (expected).",
                    len(backbone_missing),
                )
            if other_missing:
                logger.warning("SSL: unexpected missing keys: %s", other_missing[:10])
            if unexpected:
                logger.debug("SSL: %d unexpected keys in checkpoint (ignored).", len(unexpected))

            self.model.to(self.device).eval()

            # Forward hook to capture the embedding from the input to out_layer.
            def hook(module, inp, output):
                self._last_hidden = inp[0].detach()
            self.model.out_layer.register_forward_hook(hook)

        finally:
            os.chdir(cwd_backup)

        logger.info("Loaded expert ssl version=%s from %s", self.model_version, ckpt_path)

    def score(self, audio_window: np.ndarray) -> Score:
        if self.model is None:
            raise RuntimeError("SSLExpert not initialized.")

        window = np.asarray(audio_window, dtype=np.float32).reshape(-1)

        x_len = window.shape[0]
        if x_len >= _WINDOW_SAMPLES:
            window = window[:_WINDOW_SAMPLES]
        else:
            if x_len == 0:
                window = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
            else:
                num_repeats = int(_WINDOW_SAMPLES / x_len) + 1
                window = np.tile(window, (1, num_repeats))[:, :_WINDOW_SAMPLES][0]

        tensor = torch.from_numpy(window).to(self.device).unsqueeze(0)

        use_amp = self.device.type == "cuda"
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=use_amp):
            pred = self.model(tensor)

        # Model outputs [batch, 2]. Index 0 = Spoof logit, Index 1 = Bonafide logit.
        # Return Spoof − Bonafide so positive = more likely spoof.
        logit = float(pred[0, 0].item() - pred[0, 1].item())

        if self._last_hidden is not None:
            embedding = self._last_hidden.squeeze(0).float().cpu().numpy().tolist()
            self._last_hidden = None
        else:
            embedding = [0.0] * 160  # fallback if hook fires before hook is set

        return {
            "logit": logit,
            "embedding": embedding,
            "model_version": self.model_version,
        }
