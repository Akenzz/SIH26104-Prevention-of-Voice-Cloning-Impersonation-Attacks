"""
experts/ssl_antispoof.py — Adapter for TakHemlata/SSL_Anti-spoofing

Score output (same contract as others):
    {"logit": float, "embedding": list[float], "model_version": str}

Label convention (IMPORTANT):
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
    model_version = "ssl-antispoofing-takhemlata-pretrained"

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
            import model
            self.model = model.Model(None, self.device)
            
            ckpt_path = _SSL_REPO / "best_SSL_model_LA.pth"
            if not ckpt_path.exists():
                logger.info("SSL model not found locally. Downloading from Hugging Face...")
                spec = HUB_EXPERTS["ssl"]
                ckpt_path = ensure_checkpoint(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    cache_dir=self.cache_dir,
                    local_name=spec["local_name"],
                )
            
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.model.to(self.device).eval()
            
            # Forward hook to capture the embedding from the input to out_layer
            def hook(module, input, output):
                self._last_hidden = input[0].detach()
            self.model.out_layer.register_forward_hook(hook)

        finally:
            os.chdir(cwd_backup)
            
        logger.info(f"Loaded expert ssl version={self.model_version} from {ckpt_path}")

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
            
        # Model outputs [batch, 2]. Index 0 is Spoof, Index 1 is Bonafide (REAL).
        # We return Spoof logit minus Bonafide logit so that positive = Spoof.
        logit = float(pred[0, 0].item() - pred[0, 1].item())
        
        if self._last_hidden is not None:
            embedding = self._last_hidden.squeeze(0).float().cpu().numpy().tolist()
            self._last_hidden = None
        else:
            embedding = [0.0] * 160 # Fallback dimension if hook fails
            
        return {
            "logit": logit,
            "embedding": embedding,
            "model_version": self.model_version
        }
