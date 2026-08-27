from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger("realtime_backend.experts")


def ensure_checkpoint(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    local_name: str | None = None,
) -> Path:
    """Download a public Hub file once, then reuse the local copy.

    No token is used. Files land under ``cache_dir`` and must stay gitignored.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_name = local_name or filename
    dest = cache_dir / dest_name
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Checkpoint already cached, skipping download: %s", dest)
        return dest

    logger.info("Starting Hub download: %s / %s -> %s", repo_id, filename, cache_dir)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != dest.resolve():
        dest.write_bytes(downloaded_path.read_bytes())
        logger.info("Copied Hub file to cache path %s", dest)
    logger.info("Download finished: %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def _ensure_pickle_dependencies() -> None:
    """Register lightweight stand-ins for training-time modules that some
    checkpoints reference by pickle.

    The LFCC checkpoint stores a ``training.config.TrainingConfig`` dataclass
    under its ``config`` key. ``torch.load`` unpickles the whole object graph,
    so that class must be importable even though we only want the tensors in
    ``model_state_dict`` (the config is discarded). Rather than depend on the
    sibling ``lfcc-detector`` package at load time, register a minimal module so
    the object reconstructs via ``__new__`` + ``__dict__``. A real ``training``
    package already on ``sys.modules`` is never overridden.
    """
    if "training.config" in sys.modules:
        return
    training = sys.modules.get("training")
    if training is None:
        training = types.ModuleType("training")
        training.__path__ = []  # mark as a package so submodule import resolves
        sys.modules["training"] = training
    config_mod = types.ModuleType("training.config")

    class TrainingConfig:  # placeholder for unpickling only; fields via __dict__
        pass

    config_mod.TrainingConfig = TrainingConfig
    training.config = config_mod  # type: ignore[attr-defined]
    sys.modules["training.config"] = config_mod


def torch_load_checkpoint(path: Path, map_location: str = "cpu"):
    import torch

    _ensure_pickle_dependencies()
    logger.info("torch.load(%s, map_location=%s)", path, map_location)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
