"""
data_pipeline/assemble_multicorpus_dataset.py
=============================================
Build the V3 multi-corpus, multi-generator manifest for the LFCC-LCNN detector.

GOAL (see hindi_v2_summary.md §5): expose the model to many TTS/VC generators
across languages so we can MEASURE (not assume) generalization to unseen
generators. Every pool is kept 1:1 bonafide:spoof, bonafide and spoof are drawn
from the SAME corpus within a language (so the model can't shortcut on
"corpus/channel = label"), and each pool reserves a slice of generators as
held-out eval_ood.

FOUR training pools + real-world held-out:
  - Hindi        : Kathbath bonafide vs XTTS spoof   (IndicF5 held out -> eval_ood)
  - English-ASV  : VCTK bonafide     vs A01-A06 spoof (A07-A19 held out -> eval_ood)
  - English-MLAAD: en/none bonafide  vs ~50 gens      (15 gens held out -> eval_ood)
  - German-MLAAD : de/none bonafide  vs ~10 gens      (4 gens held out -> eval_ood)
  - In-the-Wild  : real-world public-figure audio, ENTIRELY eval_ood (never trained)

Splits emitted:
  train      : pools' in-domain generators, speaker/file disjoint from dev/eval
  dev        : held-out SPEAKERS, SAME generators (early stopping only)  [hi + en-asv]
  eval       : held-out SPEAKERS, SAME generators (in-domain skill)      [hi xtts]
  eval_ood   : held-out GENERATORS + real-world (the generalization gap) [all 5 groups]

Outputs two manifests (Hindi is already 16k-processed on disk, so it is passed
through untouched; everything else is raw and gets preprocess_parity'd next):
  manifests/mc_hindi_processed.csv   -- Hindi rows, paths already processed
  manifests/mc_new_raw.csv           -- ASVspoof/MLAAD/ITW rows, raw source paths

A `group` column tags each row so the evaluator can report each eval_ood family
SEPARATELY (per the claims-ledger rule: never merge in-domain and held-out).

Deterministic: fixed seed, so re-running reproduces the exact same split.
"""

import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
MANIFEST_DIR = repo_root / "data_pipeline" / "manifests"

SEED = 1337

# ---- corpus locations (verified on disk) ----------------------------------
HINDI_PROCESSED_CSV = MANIFEST_DIR / "hindi_dataset_processed.csv"
ASV_ROOT   = Path(r"D:\DatasetSIH\LA")
MLAAD_ROOT = Path(r"E:\DatasetSIH\mlaad")
ITW_ROOT   = Path(r"E:\DatasetSIH\in_the_wild\release_in_the_wild")

# ---- per-pool 1:1 target sizes --------------------------------------------
HI_TRAIN   = 2000      # per class
HI_EVAL    = 600       # per class (in-domain XTTS, held-out speakers)
ASV_TRAIN  = 2000
ASV_DEV    = 1000
ASV_OOD    = 1500
MLEN_TRAIN = 3000      # the skew: MLAAD contributes the most generators
MLEN_OOD   = 1500
MLDE_TRAIN = 900
MLDE_OOD   = 400
ITW_OOD    = 1500

# ---- held-out generators (fixed, reproducible, span old + modern) ---------
HELD_EN = {
    "griffin_lim", "f5-tts", "kokoro", "Edge-TTS", "Higgs-Audio-V2",
    "OpenVoiceV2", "suno_bark", "parler_tts_mini_v1", "WhisperSpeech",
    "MeloTTS", "sesame_csm", "Chatterbox", "Qwen2.5-Omni", "Llasa-8B", "vixTTS",
}
HELD_DE = {"griffin_lim", "Edge-TTS", "RVC", "Higgs-Audio-V2"}

FINAL_COLS = [
    "path", "label", "split", "source_dataset", "speaker_id", "utterance_id",
    "generator_id", "language", "codec", "duration_s", "license", "consent",
    "held_out", "role", "text", "group",
]


def _safe(s: str) -> str:
    """Filesystem/id-safe token."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def _row(path, label, split, source, spk, utt, gen, lang, lic, consent, group,
         held_out="no", role="matched", text=""):
    return {
        "path": str(path), "label": label, "split": split,
        "source_dataset": source, "speaker_id": str(spk), "utterance_id": utt,
        "generator_id": gen, "language": lang, "codec": "src", "duration_s": -1.0,
        "license": lic, "consent": consent, "held_out": held_out, "role": role,
        "text": text, "group": group,
    }


def stratified_sample(df: pd.DataFrame, n: int, key: str, rng) -> pd.DataFrame:
    """Sample ~n rows spread as evenly as possible across values of `key`."""
    if len(df) <= n:
        return df.copy()
    groups = list(df.groupby(key))
    per = max(1, n // len(groups))
    picks = []
    for _, g in groups:
        take = min(per, len(g))
        picks.append(g.sample(take, random_state=rng.integers(1 << 30)))
    out = pd.concat(picks, ignore_index=True)
    if len(out) > n:                      # trim overshoot
        out = out.sample(n, random_state=rng.integers(1 << 30)).reset_index(drop=True)
    elif len(out) < n:                    # top up from the remainder
        rest = df.drop(index=pd.concat(picks).index, errors="ignore")
        if len(rest):
            extra = rest.sample(min(n - len(out), len(rest)),
                                random_state=rng.integers(1 << 30))
            out = pd.concat([out, extra], ignore_index=True)
    return out.reset_index(drop=True)


# ===========================================================================
# Pool 1 - Hindi (reuse the already-processed V2 manifest)
# ===========================================================================
def pool_hindi(rng) -> pd.DataFrame:
    df = pd.read_csv(HINDI_PROCESSED_CSV, encoding="utf-8-sig", dtype={"speaker_id": str})
    xtts    = df[df.generator_id == "coqui-xtts-v2"].copy()
    indicf5 = df[df.generator_id == "ai4bharat-indicf5"].copy()
    bona    = df[df.label == "bonafide"].copy()

    out = []

    def emit(sub, split, group, label_override=None):
        for r in sub.itertuples(index=False):
            out.append(_row(
                r.path, label_override or r.label, split, "Kathbath+XTTS",
                r.speaker_id, f"hi_{_safe(r.utterance_id)}_{_safe(r.generator_id)}"
                if r.label == "spoof" else f"hi_{_safe(r.utterance_id)}_bona",
                r.generator_id if r.label == "spoof" else "none", "hi",
                "CPML-noncommercial", "yes", group,
                text=getattr(r, "text", "") if isinstance(getattr(r, "text", ""), str) else "",
            ))

    # TRAIN: XTTS spoof + bonafide (both from the train split -> disjoint speakers vs dev/eval)
    emit(stratified_sample(xtts[xtts.split == "train"], HI_TRAIN, "generator_id", rng), "train", "hi_train")
    emit(bona[bona.split == "train"].sample(min(HI_TRAIN, (bona.split == "train").sum()),
                                             random_state=SEED), "train", "hi_train")
    # DEV: XTTS dev
    emit(xtts[xtts.split == "dev"], "dev", "hi_dev")
    emit(bona[bona.split == "dev"], "dev", "hi_dev")
    # EVAL (in-domain, held-out speakers): 600 XTTS + 600 bonafide (first slice)
    eval_bona = bona[bona.split == "eval"].reset_index(drop=True)
    emit(xtts[xtts.split == "eval"], "eval", "indomain_hi_xtts")
    emit(eval_bona.iloc[:HI_EVAL], "eval", "indomain_hi_xtts")
    # EVAL_OOD: IndicF5 (unseen generator) paired with a DISJOINT slice of eval bonafide
    n_if5 = len(indicf5)
    emit(indicf5, "eval_ood", "ood_hi_indicf5")
    ood_bona = eval_bona.iloc[HI_EVAL:HI_EVAL + n_if5]
    emit(ood_bona, "eval_ood", "ood_hi_indicf5")

    res = pd.DataFrame(out)
    res["held_out"] = np.where(res.group.str.startswith("ood_"), "yes", "no")
    return res


# ===========================================================================
# Pool 2 - English / ASVspoof2019 LA  (A01-A06 train, A07-A19 held out)
# ===========================================================================
def _parse_asv(protocol: Path):
    rows = []
    with open(protocol, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            rows.append((p[0], p[1], p[3], p[4]))   # spk, utt, sys, key
    return pd.DataFrame(rows, columns=["speaker_id", "utt", "sys", "key"])


def pool_asvspoof(rng) -> pd.DataFrame:
    proto = ASV_ROOT / "ASVspoof2019_LA_cm_protocols"
    specs = [
        ("train", proto / "ASVspoof2019.LA.cm.train.trn.txt", ASV_ROOT / "ASVspoof2019_LA_train" / "flac"),
        ("dev",   proto / "ASVspoof2019.LA.cm.dev.trl.txt",   ASV_ROOT / "ASVspoof2019_LA_dev" / "flac"),
        ("eval",  proto / "ASVspoof2019.LA.cm.eval.trl.txt",  ASV_ROOT / "ASVspoof2019_LA_eval" / "flac"),
    ]
    out = []

    def emit(sub, flac_dir, split, group):
        for r in sub.itertuples(index=False):
            lab = "bonafide" if r.key == "bonafide" else "spoof"
            out.append(_row(
                flac_dir / f"{r.utt}.flac", lab, split, "ASVspoof2019_LA",
                r.speaker_id, f"asv_{r.utt}",
                "none" if lab == "bonafide" else r.sys, "en", "ASVspoof_EULA", "yes",
                group, held_out=("yes" if split == "eval_ood" else "no"),
            ))

    for name, protocol, flac_dir in specs:
        df = _parse_asv(protocol)
        bona = df[df.key == "bonafide"]
        spoof = df[df.key == "spoof"]
        if name == "train":
            emit(bona.sample(min(ASV_TRAIN, len(bona)), random_state=SEED), flac_dir, "train", "asv_train")
            emit(stratified_sample(spoof, ASV_TRAIN, "sys", rng), flac_dir, "train", "asv_train")
        elif name == "dev":
            emit(bona.sample(min(ASV_DEV, len(bona)), random_state=SEED), flac_dir, "dev", "asv_dev")
            emit(stratified_sample(spoof, ASV_DEV, "sys", rng), flac_dir, "dev", "asv_dev")
        else:  # eval -> A07-A19 unseen generators -> eval_ood
            emit(bona.sample(min(ASV_OOD, len(bona)), random_state=SEED), flac_dir, "eval_ood", "ood_en_asv")
            emit(stratified_sample(spoof, ASV_OOD, "sys", rng), flac_dir, "eval_ood", "ood_en_asv")

    return pd.DataFrame(out)


# ===========================================================================
# Pools 3 & 4 - MLAAD  (mlaad/<lang>/<gen>/*.wav ; `none` = bonafide)
# ===========================================================================
def _list_wavs(folder: Path):
    try:
        return [folder / f for f in os.listdir(folder) if f.lower().endswith((".wav", ".flac"))]
    except FileNotFoundError:
        return []


def pool_mlaad(lang: str, held: set, n_train: int, n_ood: int, rng) -> pd.DataFrame:
    root = MLAAD_ROOT / lang
    gens = [d for d in os.listdir(root) if (root / d).is_dir()]
    spoof_gens = [g for g in gens if g != "none"]
    train_gens = [g for g in spoof_gens if g not in held]
    ood_gens = [g for g in spoof_gens if g in held]
    lic = "CC-BY-NC-4.0"

    def spoof_rows(gen_list, target, split, group):
        rows = []
        for g in gen_list:
            for w in _list_wavs(root / g):
                rows.append((g, w))
        gdf = pd.DataFrame(rows, columns=["gen", "wav"])
        if len(gdf) == 0:
            return []
        picked = stratified_sample(gdf, target, "gen", rng)
        return [_row(
            r.wav, "spoof", split, f"MLAAD-{lang}", f"mlaad{lang}_{_safe(r.gen)}",
            f"ml{lang}_{_safe(r.gen)}_{_safe(Path(r.wav).stem)}", r.gen, lang, lic, "no",
            group, held_out=("yes" if split == "eval_ood" else "no"),
        ) for r in picked.itertuples(index=False)]

    def bona_rows(wavs, split, group):
        return [_row(
            w, "bonafide", split, f"MLAAD-{lang}", f"mlaad{lang}_none",
            f"ml{lang}_none_{_safe(w.stem)}", "none", lang, lic, "no",
            group, held_out=("yes" if split == "eval_ood" else "no"),
        ) for w in wavs]

    # bonafide `none`: reserve OOD slice first, then train slice (DISJOINT files)
    none_wavs = _list_wavs(root / "none")
    rng2 = np.random.default_rng(SEED + (1 if lang == "en" else 2))
    idx = rng2.permutation(len(none_wavs))
    ood_bona_w = [none_wavs[i] for i in idx[:n_ood]]
    train_bona_w = [none_wavs[i] for i in idx[n_ood:n_ood + n_train]]

    out = []
    # Balance each split 1:1 by trimming the LARGER class down to the smaller
    # (German `none` is thin, so bonafide can be the limiting side).
    def add_balanced(spoof, bona, tag):
        k = min(len(spoof), len(bona))
        if len(spoof) != len(bona):
            print(f"  [mlaad-{lang}] {tag}: spoof={len(spoof)} bona={len(bona)} -> balanced to {k}/{k}")
        out.extend(spoof[:k]);  out.extend(bona[:k])

    add_balanced(spoof_rows(train_gens, n_train, "train", f"ml{lang}_train"),
                 bona_rows(train_bona_w, "train", f"ml{lang}_train"), "train")
    add_balanced(spoof_rows(ood_gens, n_ood, "eval_ood", f"ood_{lang}_mlaad"),
                 bona_rows(ood_bona_w, "eval_ood", f"ood_{lang}_mlaad"), "eval_ood")

    res = pd.DataFrame(out)
    res.attrs["train_gens"] = train_gens
    res.attrs["ood_gens"] = ood_gens
    return res


# ===========================================================================
# Pool 5 - In-the-Wild  (entirely eval_ood; real-world public-figure audio)
# ===========================================================================
def pool_itw(rng) -> pd.DataFrame:
    meta = pd.read_csv(ITW_ROOT / "meta.csv")
    meta.columns = [c.strip().lower() for c in meta.columns]
    def norm(v):
        v = str(v).strip().lower()
        return "bonafide" if v in ("bona-fide", "bonafide", "real", "genuine") else "spoof"
    meta["lab"] = meta["label"].map(norm)
    out = []
    for lab in ("bonafide", "spoof"):
        sub = meta[meta.lab == lab]
        sub = sub.sample(min(ITW_OOD, len(sub)), random_state=SEED)
        for r in sub.itertuples(index=False):
            out.append(_row(
                ITW_ROOT / r.file, lab, "eval_ood", "In-the-Wild",
                _safe(r.speaker), f"itw_{_safe(Path(r.file).stem)}",
                "none" if lab == "bonafide" else "itw-unknown", "en",
                "CC-BY-4.0", "research", "ood_itw", held_out="yes",
            ))
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame, title: str):
    print(f"\n===== {title} =====")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\nby group x label:")
    print(df.groupby(["group", "label"]).size().unstack(fill_value=0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-hindi", default=str(MANIFEST_DIR / "mc_hindi_processed.csv"))
    ap.add_argument("--out-new", default=str(MANIFEST_DIR / "mc_new_raw.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print("Building multi-corpus manifest (seed=%d)..." % SEED)

    hi = pool_hindi(rng);                       print("[hindi]   rows:", len(hi))
    asv = pool_asvspoof(rng);                   print("[asv]     rows:", len(asv))
    mlen = pool_mlaad("en", HELD_EN, MLEN_TRAIN, MLEN_OOD, rng)
    print("[mlaad-en] rows:", len(mlen), "| train_gens:", len(mlen.attrs.get("train_gens", [])),
          "| held:", sorted(mlen.attrs.get("ood_gens", [])))
    mlde = pool_mlaad("de", HELD_DE, MLDE_TRAIN, MLDE_OOD, rng)
    print("[mlaad-de] rows:", len(mlde), "| train_gens:", len(mlde.attrs.get("train_gens", [])),
          "| held:", sorted(mlde.attrs.get("ood_gens", [])))
    itw = pool_itw(rng);                        print("[itw]     rows:", len(itw))

    # Hindi is already processed on disk -> its own manifest (passthrough).
    hi = hi[FINAL_COLS]
    Path(args.out_hindi).parent.mkdir(parents=True, exist_ok=True)
    hi.to_csv(args.out_hindi, index=False, encoding="utf-8-sig")

    new = pd.concat([asv, mlen, mlde, itw], ignore_index=True)[FINAL_COLS]
    new.to_csv(args.out_new, index=False, encoding="utf-8-sig")

    combined = pd.concat([hi, new], ignore_index=True)
    summarize(combined, "COMBINED (intended final)")
    print("\nspoof generators in TRAIN split:",
          combined[(combined.split == "train") & (combined.label == "spoof")]
          .generator_id.nunique())
    print(f"\n[OK] wrote {args.out_hindi}  ({len(hi):,} Hindi passthrough rows)")
    print(f"[OK] wrote {args.out_new}  ({len(new):,} rows to preprocess)")


if __name__ == "__main__":
    main()
