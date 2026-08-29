"""
data_pipeline/assemble_hindi_dataset.py
=======================================
Build the final Hindi manifest from the spoof audio that ACTUALLY EXISTS on disk,
pairing each spoof with its bonafide original. Adapts to partial / renamed output
and grows incrementally as you add generators.

Why this exists (vs build_hindi_manifest.py): the original assumed every job was
generated into spoof/<split>/<gen>/<utt>.wav. Reality: only XTTS ran, its files
live in spoof_XTTS/<split>/xtts/<utt>.wav, and a "spoof_indicF5" folder turned out
to be a byte-identical duplicate of XTTS. This script is driven by the files you
point it at, not by the job plan, so it can't be fooled by any of that.

It keeps the dataset BALANCED and MATCHED: a bonafide clip is included only if it
has at least one spoof (1:1 pairing), plus the eval real-only clips for false-
positive rate. When IndicF5 / RVC are ready later, just add another --spoof
folder=generator and re-run — new spoofs and their bonafide partners fold in.

Usage (XTTS only, today):
  python data_pipeline/assemble_hindi_dataset.py \
      --spoof "E:/DatasetSIH/hindi_spoof/spoof_XTTS=xtts"

Later (add generators as they arrive):
  python data_pipeline/assemble_hindi_dataset.py \
      --spoof "E:/DatasetSIH/hindi_spoof/spoof_XTTS=xtts" \
      --spoof "E:/DatasetSIH/hindi_spoof/spoof_indicf5_real=indicf5" \
      --spoof "E:/DatasetSIH/hindi_spoof/spoof_rvc=rvc"
"""

import os
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

repo_root = Path(__file__).resolve().parent.parent
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"

# short name (what you pass) -> (source_dataset label, clean generator_id, license)
GEN_META = {
    "xtts":    ("Kathbath+XTTS",    "coqui-xtts-v2",     "CPML-noncommercial"),
    "indicf5": ("Kathbath+IndicF5", "ai4bharat-indicf5", "research-only"),
    "rvc":     ("Kathbath+RVC",     "rvc-consented",     "consented-teammate"),
}


def measure(path: Path):
    try:
        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


def scan_wavs(folder: Path):
    for dp, _dn, fn in os.walk(folder):
        for f in fn:
            if f.lower().endswith(".wav"):
                yield Path(dp) / f


def main():
    ap = argparse.ArgumentParser(description="Assemble the Hindi manifest from spoof audio present on disk.")
    ap.add_argument("--spoof", action="append", required=True,
                    help='repeatable "FOLDER=generator", e.g. "E:/.../spoof_XTTS=xtts"')
    ap.add_argument("--bonafide", default=str(MANIFEST_DIR / "hindi_bonafide_selected.csv"),
                    help="source of truth for bonafide rows + per-utt split/text/speaker")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "hindi_dataset.csv"))
    ap.add_argument("--held-out-gens", nargs="*", default=["rvc"])
    ap.add_argument("--no-realonly", action="store_true", help="drop the eval real-only bonafide clips")
    ap.add_argument("--all-bonafide", action="store_true",
                    help="also keep bonafide clips that have NO spoof partner (unbalanced; default off)")
    args = ap.parse_args()

    bona = pd.read_csv(args.bonafide, encoding="utf-8-sig", dtype={"speaker_id": str})
    bidx = {r.utterance_id: r for r in bona.itertuples(index=False)}
    held = set(args.held_out_gens)

    spoof_rows = []
    seen = set()          # (utt, gen) dedup
    for spec in args.spoof:
        if "=" not in spec:
            raise SystemExit(f"--spoof must be FOLDER=generator, got: {spec}")
        folder_s, gen = spec.rsplit("=", 1)
        folder, gen = Path(folder_s), gen.strip().lower()
        if gen not in GEN_META:
            print(f"[WARN] unknown generator '{gen}' — using generic metadata")
        ds, gen_id, lic = GEN_META.get(gen, (f"Kathbath+{gen}", gen, "unknown"))
        if not folder.is_dir():
            print(f"[WARN] folder not found, skipping: {folder}")
            continue

        n_used = n_stray = n_dup = 0
        for wav in scan_wavs(folder):
            utt = wav.stem
            b = bidx.get(utt)
            if b is None:                 # file not in the plan -> ignore
                n_stray += 1
                continue
            key = (utt, gen)
            if key in seen:               # same clip+generator twice -> ignore
                n_dup += 1
                continue
            seen.add(key)
            spoof_rows.append({
                "path": str(wav.resolve()), "label": "spoof", "split": b.split,
                "source_dataset": ds, "speaker_id": b.speaker_id, "utterance_id": utt,
                "generator_id": gen_id, "language": "hi", "codec": "pcm_16k",
                "duration_s": measure(wav), "license": lic, "consent": "yes",
                "held_out": "yes" if gen in held else "no",
                "role": "matched", "text": getattr(b, "text", ""),
            })
            n_used += 1
        print(f"[spoof] {folder.name} ({gen}): used={n_used:,} stray={n_stray:,} dup={n_dup:,}")

    spoof = pd.DataFrame(spoof_rows)
    if not len(spoof):
        raise SystemExit("[FAIL] no spoof files matched the bonafide plan — check --spoof folders.")
    utts_with_spoof = set(spoof.utterance_id)

    # bonafide: matched partners of real spoofs + eval real-only (FPR), unless disabled
    keep = []
    for r in bona.itertuples(index=False):
        role = getattr(r, "role", "matched")
        if role == "realonly":
            if not args.no_realonly:
                keep.append(r._asdict())
        else:  # matched
            if args.all_bonafide or r.utterance_id in utts_with_spoof:
                keep.append(r._asdict())
    bona_keep = pd.DataFrame(keep)

    full = pd.concat([bona_keep, spoof], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\n[OK] wrote {args.out}")
    print(f"  bonafide={len(bona_keep):,}  spoof={len(spoof):,}  total={len(full):,}")
    print("\nlabel x split:")
    print(full.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\nspoof generator x split:")
    print(spoof.groupby(["generator_id", "split"]).size().unstack(fill_value=0).to_string())
    # pairing sanity: matched bonafide should equal spoofed-utt count per split
    m = bona_keep[bona_keep.get("role", "matched") == "matched"] if "role" in bona_keep else bona_keep
    print(f"\nmatched bonafide={len(m):,}  unique spoofed utts={len(utts_with_spoof):,}  "
          f"(equal => clean 1:1 pairing)")


if __name__ == "__main__":
    main()
