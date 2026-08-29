"""
data_pipeline/build_hindi_manifest.py
=====================================
Merge the selected bonafide clips + the spoofs generated on Colab into the final
Hindi V2 manifest. Run AFTER you have downloaded the generated audio so that it
mirrors each job's `out_relpath` under --spoof-root.

Resumable & partial-friendly: any job whose audio is not on disk yet is skipped
and counted, so you can build an interim manifest after each generator finishes.

Output columns = the 12 REQUIRED_COLUMNS (schema.py) + held_out + role + text.
`held_out=yes` is stamped for the RVC (unseen) generator so check_leakage.py and
the eval harness treat it correctly.
"""

import os
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

repo_root = Path(__file__).resolve().parent.parent
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"
DEFAULT_SPOOF_ROOT = Path(os.environ.get("HINDI_SPOOF_ROOT", r"E:\DatasetSIH\hindi_spoof"))

# target_generator -> (source_dataset label, generator_id, license)
GEN_META = {
    "indicf5": ("Kathbath+IndicF5", "ai4bharat-indicf5", "research-only"),
    "xtts":    ("Kathbath+XTTS",    "coqui-xtts-v2",     "CPML-noncommercial"),
    "rvc":     ("Kathbath+RVC",     "rvc-consented",     "consented-teammate"),
}


def measure(path: Path):
    try:
        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Merge bonafide + generated spoofs into the final Hindi manifest.")
    ap.add_argument("--bonafide", default=str(MANIFEST_DIR / "hindi_bonafide_selected.csv"))
    ap.add_argument("--jobs", default=str(MANIFEST_DIR / "hindi_spoof_jobs.csv"))
    ap.add_argument("--spoof-root", default=str(DEFAULT_SPOOF_ROOT),
                    help="folder that contains the generated audio at each job's out_relpath")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "hindi_v2_full.csv"))
    ap.add_argument("--held-out-gens", nargs="*", default=["rvc"])
    args = ap.parse_args()

    bona = pd.read_csv(args.bonafide, encoding="utf-8-sig", dtype={"speaker_id": str})
    jobs = pd.read_csv(args.jobs, encoding="utf-8-sig", dtype={"speaker_id": str})
    root = Path(args.spoof_root)

    spoof_rows, missing = [], 0
    for j in jobs.itertuples(index=False):
        p = root / j.out_relpath
        if not p.exists():
            missing += 1
            continue
        ds, gen_id, lic = GEN_META.get(j.target_generator,
                                       (f"Kathbath+{j.target_generator}", j.target_generator, "unknown"))
        spoof_rows.append({
            "path": str(p.resolve()), "label": "spoof", "split": j.split,
            "source_dataset": ds, "speaker_id": j.speaker_id, "utterance_id": j.utt_id,
            "generator_id": gen_id, "language": "hi", "codec": "pcm_16k",
            "duration_s": measure(p), "license": lic, "consent": "yes",
            "held_out": "yes" if j.target_generator in args.held_out_gens else "no",
            "role": "matched", "text": j.text,
        })

    spoof = pd.DataFrame(spoof_rows)
    full = pd.concat([bona, spoof], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"bonafide={len(bona):,}  spoof={len(spoof):,}  missing_audio={missing:,}  -> {args.out}")
    if len(spoof):
        print("\nlabel x split:")
        print(full.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
        print("\nspoof generator x split:")
        print(spoof.groupby(["generator_id", "split"]).size().unstack(fill_value=0).to_string())
    if missing:
        print(f"\n[NOTE] {missing:,} jobs have no audio yet — finish generation, re-download, re-run.")


if __name__ == "__main__":
    main()
