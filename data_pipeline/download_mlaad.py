#!/usr/bin/env python3
"""
download_mlaad.py
=================
Downloads a language-filtered subset of the MLAAD "fake" audio dataset
(https://huggingface.co/datasets/mueller91/MLAAD) to a local dataset drive.

MLAAD ships ONLY spoofed/synthetic audio. Pair it with a bonafide corpus
(e.g. In-The-Wild) downstream to avoid a spoof-skewed training set.

Repo layout (source):
    fake/<language>/<model>/{meta.csv, *.wav}

FINAL LOCAL layout produced by this script (fake/ wrapper removed,
each language promoted to a top-level folder under the dataset dir):
    E:\\DatasetSIH\\MLAAD\\hi\\<model>\\{meta.csv, *.wav}
    E:\\DatasetSIH\\MLAAD\\en\\<model>\\...
    E:\\DatasetSIH\\MLAAD\\ta\\<model>\\...

This script:
  * verifies you have accepted the gated license and are authenticated,
  * lists the actual language folders present in the repo,
  * downloads ONLY the languages you ask for (default: hi en kn ml mr ta),
  * moves each language folder up one level so 'fake/' disappears,
  * resumes cleanly if interrupted (snapshot_download is idempotent),
  * prints a per-language file/size summary at the end.

------------------------------------------------------------------------
PREREQUISITES (do this once, and tell your friend to do it too):

  1. pip install "huggingface_hub[hf_transfer]>=0.24"
  2. Create a token at https://huggingface.co/settings/tokens (read scope)
  3. Open https://huggingface.co/datasets/mueller91/MLAAD and click
     "Agree and access repository"  (the dataset is GATED — downloads
     fail with 401/403 until you accept the license while logged in).
  4. Authenticate ONE of these ways:
       - run:  huggingface-cli login        (recommended), OR
       - set env var:  HF_TOKEN=hf_xxx       (this script reads it), OR
       - pass --token hf_xxx  on the command line.

------------------------------------------------------------------------
USAGE:

  # default: hi,en,kn,ml,mr,ta -> E:\\DatasetSIH\\MLAAD\\<lang>\\...
  python download_mlaad.py

  # custom languages / destination root
  python download_mlaad.py --langs hi en ta --dest "E:\\DatasetSIH\\MLAAD"

  # just list what languages the repo actually has, download nothing
  python download_mlaad.py --list-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# Faster multi-threaded transfers when hf_transfer is installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO_ID = "mueller91/MLAAD"
REPO_TYPE = "dataset"
DEFAULT_LANGS = ["hi", "en", "kn", "ml", "mr", "ta"]
DEFAULT_DEST = r"E:\DatasetSIH\MLAAD"


def die(msg: str, code: int = 1) -> None:
    print(f"\n[FATAL] {msg}\n", file=sys.stderr)
    sys.exit(code)


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} TB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download language-filtered MLAAD 'fake' subset, "
                    "flattened to <dest>/<lang>/...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS,
                   help="ISO language folder names to fetch from fake/.")
    p.add_argument("--dest", default=DEFAULT_DEST,
                   help="Destination root (created if missing). Each language "
                        "becomes a top-level folder inside it.")
    p.add_argument("--token", default=None,
                   help="HF token; defaults to HF_TOKEN env / cached login.")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent download workers.")
    p.add_argument("--list-only", action="store_true",
                   help="Only list available language folders, then exit.")
    return p.parse_args()


def get_token(cli_token: str | None) -> str | None:
    # Priority: --token > env > cached CLI login (handled by hub itself).
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def flatten_fake_wrapper(dest: Path, langs: list[str]) -> None:
    """
    snapshot_download recreates 'fake/<lang>/...' under dest.
    Promote each <lang> folder to dest/<lang> and remove the empty 'fake/'.
    Idempotent: safe to run again on a resumed download.
    """
    fake_dir = dest / "fake"
    if not fake_dir.is_dir():
        return  # already flattened (or nothing came down)
    for lang in langs:
        src = fake_dir / lang
        if not src.is_dir():
            continue
        dst = dest / lang
        if dst.exists():
            # Merge: move any files from src that aren't already at dst.
            for item in src.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(src)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.move(str(item), str(target))
            shutil.rmtree(src, ignore_errors=True)
        else:
            shutil.move(str(src), str(dst))
    # Remove 'fake/' if now empty (ignore leftover .cache/ etc. safely).
    try:
        remaining = [p for p in fake_dir.rglob("*") if p.is_file()]
        if not remaining:
            shutil.rmtree(fake_dir, ignore_errors=True)
    except FileNotFoundError:
        pass


def summarize(dest: Path, langs: list[str]) -> None:
    """Walk the flattened tree and print a per-language file/size report."""
    print("\n[5/5] Download summary")
    print("      " + "-" * 52)
    print(f"      {'language':<12}{'files':>10}{'audio':>10}{'size':>16}")
    print("      " + "-" * 52)
    grand_files = grand_audio = grand_bytes = 0
    for lang in langs:
        lang_dir = dest / lang
        if not lang_dir.is_dir():
            print(f"      {lang:<12}{'MISSING':>10}")
            continue
        n_files = n_audio = 0
        n_bytes = 0
        for f in lang_dir.rglob("*"):
            if f.is_file():
                n_files += 1
                n_bytes += f.stat().st_size
                if f.suffix.lower() in (".wav", ".flac", ".mp3"):
                    n_audio += 1
        grand_files += n_files
        grand_audio += n_audio
        grand_bytes += n_bytes
        print(f"      {lang:<12}{n_files:>10}{n_audio:>10}{human(n_bytes):>16}")
    print("      " + "-" * 52)
    print(f"      {'TOTAL':<12}{grand_files:>10}{grand_audio:>10}{human(grand_bytes):>16}")
    print("      " + "-" * 52)


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        die('huggingface_hub not installed. Run:\n'
            '    pip install "huggingface_hub[hf_transfer]>=0.24"')

    token = get_token(args.token)
    api = HfApi(token=token)

    # --- 1. Confirm access + enumerate real language folders ------------
    print(f"[1/5] Inspecting repo '{REPO_ID}' (dataset) ...")
    try:
        all_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE, token=token)
    except GatedRepoError:
        die("Repo is GATED and not yet accepted for your account.\n"
            "  -> Open https://huggingface.co/datasets/mueller91/MLAAD\n"
            "     and click 'Agree and access repository' while logged in,\n"
            "     then re-run. Also ensure you are authenticated "
            "(huggingface-cli login or HF_TOKEN).")
    except RepositoryNotFoundError:
        die(f"Repo '{REPO_ID}' not found or you lack access (check token).")
    except HfHubHTTPError as e:
        die(f"HTTP error while listing repo files: {e}\n"
            "  Most common cause: not logged in / license not accepted.")

    # Map: language -> list of files under fake/<language>/
    langs_present: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0] == "fake":
            langs_present[parts[1]].append(f)

    available = sorted(langs_present.keys())
    print(f"      Languages available under fake/: {', '.join(available)}")

    if args.list_only:
        print("\n[list-only] Done.")
        return

    requested = list(dict.fromkeys(args.langs))  # de-dup, keep order
    found = [l for l in requested if l in langs_present]
    missing = [l for l in requested if l not in langs_present]

    if missing:
        print(f"      [WARN] Requested but NOT present in repo: {', '.join(missing)}")
    if not found:
        die("None of the requested languages exist in the repo. "
            f"Available: {', '.join(available)}")

    print(f"      Will download: {', '.join(found)}")
    total_planned = sum(len(langs_present[l]) for l in found)
    print(f"      Planned files (incl. meta.csv): {total_planned}")

    # --- 2. Prepare destination ---------------------------------------
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[2/5] Destination root: {dest.resolve()}")

    # allow_patterns scoped to just the chosen languages under fake/.
    allow_patterns = [f"fake/{lang}/**" for lang in found]

    # --- 3. Download (idempotent / resumable) -------------------------
    print(f"[3/5] Starting download ({args.workers} workers, "
          f"hf_transfer={'on' if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1' else 'off'}) ...")
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(dest),
            allow_patterns=allow_patterns,
            max_workers=args.workers,
            token=token,
        )
    except GatedRepoError:
        die("Gated repo not accepted — see instructions above.")
    except HfHubHTTPError as e:
        die(f"Download failed (HTTP): {e}")
    except KeyboardInterrupt:
        die("Interrupted by user. Re-run the same command to resume.", code=130)

    elapsed = time.time() - t0
    print(f"      Download finished in {elapsed / 60:.1f} min.")

    # --- 4. Flatten: fake/<lang> -> <lang> ----------------------------
    print("[4/5] Flattening 'fake/' wrapper -> top-level language folders ...")
    flatten_fake_wrapper(dest, found)

    # --- 5. Summary ----------------------------------------------------
    summarize(dest, found)
    print("\n[DONE] MLAAD subset ready at "
          f"{dest.resolve()}\\<lang>\\...  (ALL spoof — "
          "balance with a bonafide set (ITW) before training).")


if __name__ == "__main__":
    main()