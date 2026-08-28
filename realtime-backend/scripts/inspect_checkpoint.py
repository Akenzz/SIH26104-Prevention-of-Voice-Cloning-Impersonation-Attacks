"""Download (if needed) and print Hub checkpoint structure without wiring a model.

    python scripts/inspect_checkpoint.py wavlm
    python scripts/inspect_checkpoint.py lfcc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HUB_EXPERTS, MODEL_CACHE_DIR
from experts.hub import ensure_checkpoint, torch_load_checkpoint


def summarize(obj, prefix: str = "", depth: int = 0) -> None:
    if depth > 2:
        print(f"{prefix}...")
        return
    if isinstance(obj, dict):
        print(f"{prefix}dict keys={list(obj.keys())[:20]}")
        for key in list(obj.keys())[:12]:
            summarize(obj[key], prefix + f"  {key}: ", depth + 1)
        return
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            print(f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
            return
    except Exception:
        pass
    print(f"{prefix}{type(obj).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expert", choices=sorted(HUB_EXPERTS))
    args = parser.parse_args()
    spec = HUB_EXPERTS[args.expert]
    path = ensure_checkpoint(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        cache_dir=MODEL_CACHE_DIR,
        local_name=spec["local_name"],
    )
    blob = torch_load_checkpoint(path, map_location="cpu")
    print(f"loaded {path}")
    summarize(blob)


if __name__ == "__main__":
    main()
