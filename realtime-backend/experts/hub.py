from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

logger = logging.getLogger("realtime_backend.experts")


def _remote_is_different(repo_id: str, filename: str, local: Path) -> bool:
    """True only when we can PROVE the Hub copy differs from ``local``.

    Existence-only caching is a trap: when a teammate re-uploads a checkpoint
    under the same filename, every machine that already downloaded the old one
    keeps loading stale weights forever, and a stale checkpoint paired with a
    fresh calibrator reads as confident nonsense rather than as an error.

    Cheap HEAD request. Any failure (offline, rate limit, no metadata) returns
    False so the cached file is used — being offline must never break startup.
    """
    try:
        meta = get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename=filename), timeout=10)
    except Exception as exc:
        logger.info("Could not check %s/%s for updates (%s) — using cached copy.",
                    repo_id, filename, type(exc).__name__)
        return False
    remote_size = getattr(meta, "size", None)
    if remote_size is None:
        return False
    local_size = local.stat().st_size
    if remote_size == local_size:
        return False
    logger.warning(
        "Cached checkpoint %s is STALE: local %d bytes, Hub %s/%s now %d bytes. Re-downloading.",
        local, local_size, repo_id, filename, remote_size,
    )
    return True


def ensure_checkpoint(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    local_name: str | None = None,
) -> Path:
    """Download a public Hub file once, then reuse the local copy.

    First run pulls the weights from Hugging Face; later runs load from disk and
    only make one cheap HEAD request to confirm the cached file still matches the
    Hub. No token is used — both repos are public. Files must stay gitignored.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_name = local_name or filename
    dest = cache_dir / dest_name
    if dest.exists() and dest.stat().st_size > 0:
        if not _remote_is_different(repo_id, filename, dest):
            logger.info("Checkpoint already cached, skipping download: %s", dest)
            return dest
        dest.unlink()

    logger.info(
        "Checkpoint %s not found locally — downloading from Hugging Face (%s / %s). "
        "This happens once; later runs load from disk.",
        dest,
        repo_id,
        filename,
    )
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(cache_dir),
        )
    except Exception as exc:
        # A fresh clone with no weights and no network is the single most likely
        # first-run failure; a raw traceback here tells a teammate nothing.
        raise RuntimeError(
            f"Could not obtain the model checkpoint {filename!r} for this expert.\n"
            f"  Tried local cache: {dest}\n"
            f"  Tried Hugging Face: https://huggingface.co/{repo_id} (file: {filename})\n"
            f"  Underlying error: {type(exc).__name__}: {exc}\n"
            "Fix: connect to the internet and rerun, or download the file manually "
            f"from that page and place it at {dest}."
        ) from exc
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != dest.resolve():
        dest.write_bytes(downloaded_path.read_bytes())
        logger.info("Copied Hub file to cache path %s", dest)
        # hf_hub_download writes the file under its *repo* name. When we rename
        # it (local_name differs) that original is a byte-identical duplicate —
        # 380 MB for WavLM — so drop it instead of leaving both on every
        # teammate's disk. Only ever inside our own cache_dir.
        try:
            if cache_dir.resolve() in downloaded_path.resolve().parents:
                downloaded_path.unlink()
                logger.info("Removed duplicate download %s", downloaded_path)
        except OSError as exc:
            logger.warning("Could not remove duplicate download %s: %s", downloaded_path, exc)
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
