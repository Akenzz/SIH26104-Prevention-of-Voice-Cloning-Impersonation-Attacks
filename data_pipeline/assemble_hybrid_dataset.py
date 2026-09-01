"""
data_pipeline/assemble_hybrid_dataset.py
========================================
Build the CLEAN-MODEL "hybrid" raw manifest for the LFCC-LCNN detector.

This is NOT a warm-start of mc_v3 — it is a fresh, balanced training set that
merges the friend's scale/round-robin/OOD structure with our anti-shortcut rule:

  friend's structure
    - ~20k bonafide + ~20k spoof BASE clips (VAD-sliced to ~4-10s chunks later)
    - bonafide round-robin / speaker-stratified across corpora
    - spoof round-robin / generator-stratified across corpora
    - dev = official dev slices; eval = pristine in-domain sanity set
    - eval_ood = UNSEEN generators + real-world (In-the-Wild) held-out slice

  our anti-shortcut rule (the mc_v3 lesson)
    - pair bonafide and spoof from the SAME language in every split, so the
      model can't shortcut on "language/channel = label"
    - channel augmentation is applied EQUALLY to both classes at TRAIN time
      (loader-side, via augmentation.py) — not baked here
    - In-the-Wild is used in TRAIN for English (user explicitly lifted the
      held-out-only rule); a DISJOINT speaker slice is reserved for eval_ood

Label convention (NEVER flip): bonafide=0, spoof=1.

Two-stage pattern, same as assemble_multicorpus_dataset.py:
  1. this script            -> raw manifest (source paths, 16 cols)
  2. preprocess_parity/VAD  -> 16k mono / trim / loudness / 4-10s chunks

Outputs:
  manifests/hybrid_hindi_processed.csv  -- Hindi rows already processed on disk
  manifests/hybrid_new_raw.csv          -- everything else, raw source paths

Deterministic: fixed SEED so re-running reproduces the exact same split.
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
HINDI_PROCESSED_CSV = MANIFEST_DIR / "hindi_dataset_processed.csv"   # Hindi bona + XTTS/IndicF5 spoof, already 16k
ASV_ROOT     = Path(r"D:\DatasetSIH\LA")
MLAAD_FAKE   = Path(r"E:\DatasetSIH\mlaad\fake")                     # mlaad/fake/<lang>/<gen>/*.wav  (spoof only)
ITW_ROOT     = Path(r"E:\DatasetSIH\in_the_wild\release_in_the_wild")
KATHBATH_DIR = Path(r"E:\DatasetSIH\kathbath")                       # kathbath/<language>/*.wav (bonafide)
GV_ROOT      = Path(r"E:\DatasetSIH\newhindi\GV_Train_100h")         # Gramvaani Hindi bonafide (mp3)

INDIC_DIRS = {  # kn/ml/mr/ta bonafide downloaded from Kathbath (dir name -> iso)
    "kannada": "kn", "malayalam": "ml", "marathi": "mr", "tamil": "ta",
}

# ---- balanced base targets (per class; VAD slicing multiplies later) -------
# Kept 1:1 bonafide:spoof WITHIN each language so language can't be a shortcut.
# Scaled to ~20k bonafide / ~20k spoof TRAIN total, kept 1:1 within each language.
# Hindi total = processed pool (capped ~4609 by XTTS+IndicF5) + GV raw pool (HI_TRAIN).
# Indic is spoof-limited and cannot grow past ~1500 total, so the extra volume
# comes from Hindi + English.
HI_TRAIN   = 4700     # Hindi GV/mlaad-hi pool; processed pool adds ~4609 -> Hindi ~9.3k/class
HI_EVAL    =  600     # in-domain sanity (held-out speakers)
EN_TRAIN   = 9200     # English: bona (ITW-train + ASV-bona) vs spoof (mlaad-en + ASV A01-A06 + ITW)
EN_EVAL    =  600
EN_OOD     = 1500     # eval_ood: unseen mlaad-en gens + ASV A07-A19 + disjoint ITW slice
INDIC_CAP  =  600     # per Indic lang, per class (spoof-limited: kn 600, ml/mr/ta 300 -> ~1500 total)

# English generators reserved for eval_ood (UNSEEN at train time). Preferred
# modern names if present; topped up deterministically to N_HELD_EN.
HELD_EN_PREF = {
    "Higgs-Audio-V2", "Edge-TTS", "Chatterbox", "Cartesia.ai (Sonic-3)",
    "FireRedTTS-2.0", "Index-TTS-2.0", "Llasa-8B", "Qwen2.5-Omni",
    "OpenVoiceV2", "MeloTTS",
}
N_HELD_EN = 25        # ~1/6 of the 143 en generators held out

FINAL_COLS = [
    "path", "label", "split", "source_dataset", "speaker_id", "utterance_id",
    "generator_id", "language", "codec", "duration_s", "license", "consent",
    "held_out", "role", "text", "group",
]


def _safe(s: str) -> str:
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
    """Round-robin sample ~n rows spread as evenly as possible across `key`."""
    if len(df) <= n:
        return df.copy()
    groups = list(df.groupby(key))
    per = max(1, n // len(groups))
    picks = []
    for _, g in groups:
        picks.append(g.sample(min(per, len(g)), random_state=rng.integers(1 << 30)))
    out = pd.concat(picks, ignore_index=True)
    if len(out) > n:
        out = out.sample(n, random_state=rng.integers(1 << 30)).reset_index(drop=True)
    elif len(out) < n:
        rest = df.drop(index=pd.concat(picks).index, errors="ignore")
        if len(rest):
            extra = rest.sample(min(n - len(out), len(rest)),
                                random_state=rng.integers(1 << 30))
            out = pd.concat([out, extra], ignore_index=True)
    return out.reset_index(drop=True)


def _list_audio(folder: Path):
    try:
        return [folder / f for f in os.listdir(folder)
                if f.lower().endswith((".wav", ".flac", ".mp3"))]
    except FileNotFoundError:
        return []


# ===========================================================================
# Pool 1 - Hindi PROCESSED (passthrough: already 16k on disk)
#   bona 'none' (Kathbath-hi)  vs  spoof coqui-xtts-v2 + ai4bharat-indicf5
#   Same corpus, already parity-processed -> goes to the *processed* manifest.
# ===========================================================================
def pool_hindi_processed(rng) -> pd.DataFrame:
    df = pd.read_csv(HINDI_PROCESSED_CSV, encoding="utf-8-sig", dtype={"speaker_id": str})
    out = []

    def emit(sub, split, group):
        for r in sub.itertuples(index=False):
            lab = r.label
            out.append(_row(
                r.path, lab, split, "Kathbath+XTTS+IndicF5",
                r.speaker_id,
                f"hi_{_safe(r.utterance_id)}_{_safe(r.generator_id)}" if lab == "spoof"
                else f"hi_{_safe(r.utterance_id)}_bona",
                r.generator_id if lab == "spoof" else "none", "hi",
                "CPML-noncommercial", "yes", group,
                held_out=("yes" if split == "eval_ood" else "no"),
                text=getattr(r, "text", "") if isinstance(getattr(r, "text", ""), str) else "",
            ))

    bona  = df[df.label == "bonafide"]
    spoof = df[df.label == "spoof"]

    # TRAIN: balanced XTTS+IndicF5 spoof vs Kathbath bonafide (both from train split)
    sp_tr = spoof[spoof.split == "train"]
    bo_tr = bona[bona.split == "train"]
    k = min(HI_TRAIN, len(sp_tr), len(bo_tr))
    emit(stratified_sample(sp_tr, k, "generator_id", rng), "train", "hi_train")
    emit(bo_tr.sample(k, random_state=SEED), "train", "hi_train")

    # DEV: official dev slices
    emit(spoof[spoof.split == "dev"], "dev", "hi_dev")
    emit(bona[bona.split == "dev"], "dev", "hi_dev")

    # EVAL sanity (in-domain, held-out speakers): balanced XTTS vs bonafide
    sp_ev = spoof[spoof.split == "eval"]
    bo_ev = bona[bona.split == "eval"].reset_index(drop=True)
    ke = min(HI_EVAL, len(sp_ev), len(bo_ev))
    emit(sp_ev.sample(ke, random_state=SEED), "eval", "indomain_hi")
    emit(bo_ev.iloc[:ke], "eval", "indomain_hi")

    return pd.DataFrame(out)


# ---- shared MLAAD-fake spoof lister ---------------------------------------
def _mlaad_spoof_df(lang: str) -> pd.DataFrame:
    """All spoof clips under mlaad/fake/<lang> as (gen, wav) rows."""
    root = MLAAD_FAKE / lang
    rows = []
    try:
        gens = [d for d in os.listdir(root) if (root / d).is_dir()]
    except FileNotFoundError:
        return pd.DataFrame(columns=["gen", "wav"])
    for g in gens:
        for w in _list_audio(root / g):
            rows.append((g, w))
    return pd.DataFrame(rows, columns=["gen", "wav"])


def _mlaad_rows(picked, lang, split, group, lic="CC-BY-NC-4.0"):
    return [_row(
        r.wav, "spoof", split, f"MLAAD-{lang}", f"mlaad{lang}_{_safe(r.gen)}",
        f"ml{lang}_{_safe(r.gen)}_{_safe(Path(r.wav).stem)}", r.gen, lang, lic, "no",
        group, held_out=("yes" if split == "eval_ood" else "no"),
    ) for r in picked.itertuples(index=False)]


# ===========================================================================
# Pool 2 - Hindi RAW: GV (Gramvaani) bonafide  vs  mlaad-hi spoof
#   Same language (hi), raw source paths -> *new_raw* manifest.
# ===========================================================================
def pool_hindi_raw(rng) -> pd.DataFrame:
    # spoof: all 9 mlaad-hi generators, round-robin
    sp = _mlaad_spoof_df("hi")
    # bonafide: GV mp3 (speaker ~ uttid prefix "SS-NNNNN-CC")
    gv = _list_audio(GV_ROOT / "Audio")
    gv_df = pd.DataFrame({"wav": gv})
    if len(gv_df):
        gv_df["spk"] = gv_df.wav.map(lambda p: Path(p).stem.split("-")[0])

    n = min(HI_TRAIN, len(sp), len(gv_df))
    out = []
    if n > 0:
        sp_pick = stratified_sample(sp, n, "gen", rng)
        gv_pick = stratified_sample(gv_df, n, "spk", rng)
        out += _mlaad_rows(sp_pick, "hi", "train", "hi_train")
        for r in gv_pick.itertuples(index=False):
            out.append(_row(
                r.wav, "bonafide", "train", "Gramvaani", f"gv_{_safe(r.spk)}",
                f"gv_{_safe(Path(r.wav).stem)}", "none", "hi",
                "CC-BY-4.0", "yes", "hi_train",
            ))
    return pd.DataFrame(out)


# ===========================================================================
# Pool 3 - English RAW
#   bonafide: In-the-Wild (TRAIN, user lifted the hold-out rule) + ASVspoof bona
#   spoof   : mlaad-en in-domain gens (TRAIN) + ASVspoof A01-A06 (TRAIN)
#   eval_ood: UNSEEN mlaad-en gens + ASVspoof A07-A19 + DISJOINT ITW speaker slice
# ===========================================================================
def _parse_asv_eval():
    proto = ASV_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    flac = ASV_ROOT / "ASVspoof2019_LA_train" / "flac"
    rows = []
    with open(proto, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append((p[0], p[1], p[3], p[4]))
    return pd.DataFrame(rows, columns=["spk", "utt", "sys", "key"]), flac


def pool_english_raw(rng) -> pd.DataFrame:
    """English pool, kept STRICTLY 1:1 bonafide:spoof in every split."""
    out = []

    # ---- MLAAD-en: reserve unseen generators for eval_ood -------------------
    sp = _mlaad_spoof_df("en")
    all_gens = sorted(sp.gen.unique().tolist())
    held = set(g for g in all_gens if g in HELD_EN_PREF)
    remaining = [g for g in all_gens if g not in held]
    rng_h = np.random.default_rng(SEED)
    if len(held) < N_HELD_EN and remaining:
        extra = rng_h.choice(remaining, size=min(N_HELD_EN - len(held), len(remaining)),
                             replace=False)
        held |= set(extra.tolist())
    train_gens = [g for g in all_gens if g not in held]
    sp_train_en = sp[sp.gen.isin(train_gens)]
    sp_ood_en   = sp[sp.gen.isin(held)]

    # ---- ASVspoof2019 LA train protocol ------------------------------------
    asv, flac = _parse_asv_eval()
    asv_bona  = asv[asv.key == "bonafide"]
    asv_spoof = asv[asv.key == "spoof"]

    def asv_row(r, lab, split, group):
        return _row(flac / f"{r.utt}.flac", lab, split, "ASVspoof2019_LA",
                    f"asv_{r.spk}", f"asv_{r.utt}",
                    "none" if lab == "bonafide" else r.sys, "en",
                    "ASVspoof_EULA", "yes", group,
                    held_out=("yes" if split == "eval_ood" else "no"))

    # ---- In-the-Wild: split SPEAKERS disjointly train vs eval_ood ----------
    meta = pd.read_csv(ITW_ROOT / "meta.csv")
    meta.columns = [c.strip().lower() for c in meta.columns]
    def norm(v):
        v = str(v).strip().lower()
        return "bonafide" if v in ("bona-fide", "bonafide", "real", "genuine") else "spoof"
    meta["lab"] = meta["label"].map(norm)
    spk_all = sorted(meta.speaker.astype(str).unique().tolist())
    rng_s = np.random.default_rng(SEED + 7)
    ood_spk = set(rng_s.choice(spk_all, size=max(1, len(spk_all) // 5), replace=False).tolist())
    itw_train = meta[~meta.speaker.astype(str).isin(ood_spk)]
    itw_ood   = meta[meta.speaker.astype(str).isin(ood_spk)]

    def itw_row(r, split, group):
        return _row(ITW_ROOT / r.file, r.lab, split, "In-the-Wild",
                    _safe(r.speaker), f"itw_{_safe(Path(r.file).stem)}",
                    "none" if r.lab == "bonafide" else "itw-unknown", "en",
                    "CC-BY-4.0", "research", group,
                    held_out=("yes" if split == "eval_ood" else "no"))

    def take(df, n, group, fn):
        return [fn(r, group) for r in df.sample(min(n, len(df)), random_state=SEED).itertuples(index=False)]

    # ===== TRAIN (1:1) ======================================================
    # spoof = 50% mlaad-en in-domain + 25% ASV A01-A06 + 25% ITW-train spoof
    n_ml, n_asv, n_itw = EN_TRAIN // 2, EN_TRAIN // 4, EN_TRAIN - EN_TRAIN // 2 - EN_TRAIN // 4
    out += _mlaad_rows(stratified_sample(sp_train_en, n_ml, "gen", rng), "en", "train", "en_train")
    for r in stratified_sample(asv_spoof, n_asv, "sys", rng).itertuples(index=False):
        out.append(asv_row(r, "spoof", "train", "en_train"))
    itw_tr_sp = itw_train[itw_train.lab == "spoof"]
    out += [itw_row(r, "train", "en_train") for r in
            itw_tr_sp.sample(min(n_itw, len(itw_tr_sp)), random_state=SEED).itertuples(index=False)]
    # bonafide = ALL ASV bona + ITW-train bona to fill to EN_TRAIN
    n_asv_bo = min(len(asv_bona), EN_TRAIN)
    for r in asv_bona.sample(n_asv_bo, random_state=SEED).itertuples(index=False):
        out.append(asv_row(r, "bonafide", "train", "en_train"))
    itw_tr_bo = itw_train[itw_train.lab == "bonafide"]
    out += [itw_row(r, "train", "en_train") for r in
            itw_tr_bo.sample(min(EN_TRAIN - n_asv_bo, len(itw_tr_bo)), random_state=SEED).itertuples(index=False)]

    # ===== EVAL_OOD (per-group 1:1; reported SEPARATELY) ====================
    itw_ood_bo = itw_ood[itw_ood.lab == "bonafide"].sample(frac=1.0, random_state=SEED)
    itw_ood_sp = itw_ood[itw_ood.lab == "spoof"]
    half = len(itw_ood_bo) // 2
    ref_bo, itw_bo = itw_ood_bo.iloc[:half], itw_ood_bo.iloc[half:]   # disjoint bona slices
    # (a) unseen mlaad-en generators, paired with pristine held-out real speech
    out += _mlaad_rows(stratified_sample(sp_ood_en, EN_OOD, "gen", rng),
                       "en", "eval_ood", "ood_en_mlaad")
    out += [itw_row(r, "eval_ood", "ood_en_mlaad") for r in
            ref_bo.head(EN_OOD).itertuples(index=False)]
    # (b) real-world In-the-Wild (held-out speakers), both classes
    k_itw = min(EN_OOD, len(itw_bo), len(itw_ood_sp))
    out += [itw_row(r, "eval_ood", "ood_itw") for r in itw_bo.head(k_itw).itertuples(index=False)]
    out += [itw_row(r, "eval_ood", "ood_itw") for r in
            itw_ood_sp.sample(k_itw, random_state=SEED).itertuples(index=False)]

    return pd.DataFrame(out)


# ===========================================================================
# Pool 4 - Indic (kn/ml/mr/ta) RAW
#   bonafide: Kathbath <language>/*.wav   vs   spoof mlaad/fake/<iso>
#   Same language, 1:1 balanced (spoof is the limiting side).
# ===========================================================================
def pool_indic_raw(rng) -> pd.DataFrame:
    out = []
    for dirname, iso in INDIC_DIRS.items():
        bona = _list_audio(KATHBATH_DIR / dirname)
        sp = _mlaad_spoof_df(iso)
        if not bona or len(sp) == 0:
            print(f"  [indic-{iso}] SKIP (bona={len(bona)} spoof={len(sp)})")
            continue
        bo_df = pd.DataFrame({"wav": bona})
        bo_df["spk"] = bo_df.wav.map(lambda p: (Path(p).stem.split("-") + ["0"])[1])
        k = min(INDIC_CAP, len(bona), len(sp))
        ke = max(1, k // 6)
        sp_pick = stratified_sample(sp, k, "gen", rng)
        bo_pick = stratified_sample(bo_df, k, "spk", rng)
        # spoof has no speaker -> row-slice; bonafide -> hold out whole SPEAKERS
        sp_ev, sp_tr = sp_pick.iloc[:ke], sp_pick.iloc[ke:]
        spk_order = list(bo_pick.groupby("spk").size().sort_values().index)  # smallest first
        ev_spk, acc = set(), 0
        for s in spk_order:                       # accumulate speakers until ~ke eval clips
            if acc >= ke:
                break
            ev_spk.add(s); acc += int((bo_pick.spk == s).sum())
        bo_ev = bo_pick[bo_pick.spk.isin(ev_spk)]
        bo_tr = bo_pick[~bo_pick.spk.isin(ev_spk)]
        out += _mlaad_rows(sp_tr, iso, "train", f"{iso}_train")
        out += _mlaad_rows(sp_ev, iso, "eval", f"indomain_{iso}")
        for sub, split, grp in [(bo_tr, "train", f"{iso}_train"),
                                (bo_ev, "eval", f"indomain_{iso}")]:
            for r in sub.itertuples(index=False):
                out.append(_row(
                    r.wav, "bonafide", split, "Kathbath", f"kb{iso}_{_safe(r.spk)}",
                    f"kb{iso}_{_safe(Path(r.wav).stem)}", "none", iso,
                    "CC-BY-4.0", "yes", grp))
        print(f"  [indic-{iso}] {k}/{k} clips; eval speakers={len(ev_spk)} "
              f"(train bona {len(bo_tr)}, eval bona {len(bo_ev)})")
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame, title: str):
    print(f"\n===== {title} =====")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\nby language x label:")
    print(df.groupby(["language", "label"]).size().unstack(fill_value=0).to_string())
    print("\nby group x label:")
    print(df.groupby(["group", "label"]).size().unstack(fill_value=0).to_string())
    ntr = df[(df.split == "train") & (df.label == "spoof")].generator_id.nunique()
    print(f"\nspoof generators in TRAIN split: {ntr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-hindi", default=str(MANIFEST_DIR / "hybrid_hindi_processed.csv"))
    ap.add_argument("--out-new", default=str(MANIFEST_DIR / "hybrid_new_raw.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print(f"Building HYBRID clean-model manifest (seed={SEED})...\n")

    hi_p = pool_hindi_processed(rng); print("[hindi-processed] rows:", len(hi_p))
    hi_r = pool_hindi_raw(rng);       print("[hindi-raw/GV]    rows:", len(hi_r))
    en_r = pool_english_raw(rng);     print("[english-raw]     rows:", len(en_r))
    in_r = pool_indic_raw(rng);       print("[indic-raw]       rows:", len(in_r))

    hi_p = hi_p[FINAL_COLS]
    Path(args.out_hindi).parent.mkdir(parents=True, exist_ok=True)
    hi_p.to_csv(args.out_hindi, index=False, encoding="utf-8-sig")

    new = pd.concat([hi_r, en_r, in_r], ignore_index=True)[FINAL_COLS]
    new.to_csv(args.out_new, index=False, encoding="utf-8-sig")

    combined = pd.concat([hi_p, new], ignore_index=True)
    summarize(combined, "COMBINED (intended final, pre-VAD base clips)")
    print(f"\n[OK] wrote {args.out_hindi}  ({len(hi_p):,} Hindi passthrough rows)")
    print(f"[OK] wrote {args.out_new}  ({len(new):,} raw rows to preprocess)")


if __name__ == "__main__":
    main()




