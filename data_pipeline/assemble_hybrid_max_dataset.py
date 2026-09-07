"""
data_pipeline/assemble_hybrid_max_dataset.py
============================================
Build the MAX-VOLUME "hybrid_max" raw manifest for the LFCC-LCNN detector.

Successor to assemble_hybrid_dataset.py (which produced the corpus behind the
shipped `hybrid` / `hybrid_br` checkpoints). Two deliberate changes:

  1. mlaad/fake contributes ONLY `en`, `hi` and `kn`.
     ml / mr / ta are dropped: each holds exactly ONE generator and 300 clips,
     so a ~250-clip spoof side drawn from a single TTS engine teaches "this
     engine" rather than "synthetic" -- the shortcut this project keeps paying
     for. Their Kathbath bonafide is dropped with them to keep every language
     1:1; a language present on only one side IS a label proxy.
  2. Every language is taken to the MAXIMUM the corpus allows instead of the
     previous fixed caps, still 1:1 bonafide:spoof WITHIN each language.

Label convention (NEVER flip): bonafide=0, spoof=1.

Two-stage pattern, unchanged:
  1. this script            -> raw manifest (source paths, 16 cols)
  2. preprocess_vad_slice   -> 16k mono / trim / loudness / 4-10s chunks

BANDWIDTH WARNING: stage 2 resamples with librosa, which brickwalls ~7.9 kHz
while the realtime backend's scipy resampler does not. Train this corpus with
band_gate_hz=7000 or the near-Nyquist shortcut comes straight back -- see
CALIBRATION-AND-RESAMPLING.md sections 1 and 5.

Outputs:
  manifests/hybrid_max_hindi_processed.csv  -- Hindi rows already 16k on disk
  manifests/hybrid_max_new_raw.csv          -- everything else, raw source paths

Deterministic: fixed SEED so re-running reproduces the exact same split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Reuse the shipped builder's helpers verbatim so the two corpora cannot drift
# apart in row schema, generator hold-out logic or sampling behaviour.
from data_pipeline.assemble_hybrid_dataset import (  # noqa: E402
    FINAL_COLS,
    GV_ROOT,
    HELD_EN_PREF,
    HINDI_PROCESSED_CSV,
    ITW_ROOT,
    KATHBATH_DIR,
    MANIFEST_DIR,
    N_HELD_EN,
    SEED,
    _list_audio,
    _mlaad_rows,
    _mlaad_spoof_df,
    _parse_asv_eval,
    _row,
    _safe,
    stratified_sample,
)

# ---- volume policy ---------------------------------------------------------
# TRAIN is uncapped: each language grows until its scarcer side runs out.
# The two numbers below are held-out MEASUREMENT sets, deliberately bounded --
# they cost VAD preprocessing time and buy nothing past the point where the
# per-generator recall estimate is stable.
HI_EVAL       = 600    # in-domain sanity; spoof-limited by the processed eval split
KN_EVAL_DIV   = 6      # hold out 1/6 of the Kannada pool, by SPEAKER
EN_OOD_SPOOF  = 5000   # 25 unseen generators -> 200 clips each (image plan: 60)

INDIC_KEEP = {"kannada": "kn"}   # ml/mr/ta intentionally absent -- see docstring

# ===========================================================================
# Pool 1 - Hindi PROCESSED (passthrough: already 16k on disk)
#   bona Kathbath-hi  vs  spoof coqui-xtts-v2 + ai4bharat-indicf5
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

    bona, spoof = df[df.label == "bonafide"], df[df.label == "spoof"]

    # TRAIN: uncapped -- take min(bona, spoof) of the official train split
    sp_tr, bo_tr = spoof[spoof.split == "train"], bona[bona.split == "train"]
    k = min(len(sp_tr), len(bo_tr))
    emit(stratified_sample(sp_tr, k, "generator_id", rng), "train", "hi_train")
    emit(bo_tr.sample(k, random_state=SEED), "train", "hi_train")

    # DEV: official dev slices, untouched (early-stop set)
    emit(spoof[spoof.split == "dev"], "dev", "hi_dev")
    emit(bona[bona.split == "dev"], "dev", "hi_dev")

    # EVAL sanity: in-domain, held-out speakers, 1:1
    sp_ev = spoof[spoof.split == "eval"]
    bo_ev = bona[bona.split == "eval"].reset_index(drop=True)
    ke = min(HI_EVAL, len(sp_ev), len(bo_ev))
    emit(sp_ev.sample(ke, random_state=SEED), "eval", "indomain_hi")
    emit(bo_ev.iloc[:ke], "eval", "indomain_hi")

    return pd.DataFrame(out)


# ===========================================================================
# Pool 2 - Hindi RAW: Gramvaani bonafide  vs  mlaad/fake/hi spoof (9 gens)
# ===========================================================================
def pool_hindi_raw(rng) -> pd.DataFrame:
    sp = _mlaad_spoof_df("hi")
    gv = _list_audio(GV_ROOT / "Audio")
    gv_df = pd.DataFrame({"wav": gv})
    if len(gv_df):
        gv_df["spk"] = gv_df.wav.map(lambda p: Path(p).stem.split("-")[0])

    n = min(len(sp), len(gv_df))        # uncapped: mlaad-hi is the scarcer side
    out = []
    if n > 0:
        out += _mlaad_rows(stratified_sample(sp, n, "gen", rng), "hi", "train", "hi_train")
        for r in stratified_sample(gv_df, n, "spk", rng).itertuples(index=False):
            out.append(_row(
                r.wav, "bonafide", "train", "Gramvaani", f"gv_{_safe(r.spk)}",
                f"gv_{_safe(Path(r.wav).stem)}", "none", "hi",
                "CC-BY-4.0", "yes", "hi_train",
            ))
    print(f"  [hindi-raw] mlaad-hi spoof={len(sp)} GV bona={len(gv_df)} -> {n}/class")
    return pd.DataFrame(out)


# ===========================================================================
# Pool 3 - English RAW
#   bonafide: In-the-Wild (train speakers) + ASVspoof19 bonafide
#   spoof   : mlaad/fake/en in-domain gens + ASVspoof A01-A06 + ITW-train spoof
#   eval_ood: 25 UNSEEN mlaad-en gens (vs pristine ITW real) + disjoint ITW slice
#
#   English is BONAFIDE-limited, so EN_TRAIN is derived, not chosen.
# ===========================================================================
def pool_english_raw(rng) -> pd.DataFrame:
    out = []

    # ---- mlaad-en: reserve N_HELD_EN generators for eval_ood ----------------
    # A generator name that also appears under another language would be SEEN in
    # training via that language, so it cannot honestly be called unseen. Only
    # `Edge-TTS` collides today (mlaad/fake/kn), but the shipped corpus held it
    # out anyway -- so its "27 unseen generators" was really 26 plus Edge-TTS.
    sp = _mlaad_spoof_df("en")
    reserved = set()
    for iso in INDIC_KEEP.values():
        reserved |= set(_mlaad_spoof_df(iso).gen.unique())
    reserved |= set(_mlaad_spoof_df("hi").gen.unique())

    all_gens = sorted(sp.gen.unique().tolist())
    eligible = [g for g in all_gens if g not in reserved]
    held = {g for g in eligible if g in HELD_EN_PREF}
    if HELD_EN_PREF - set(eligible):
        print(f"  [english] NOT held out (trained on via another language): "
              f"{sorted(HELD_EN_PREF - set(eligible))}")
    remaining = [g for g in eligible if g not in held]
    rng_h = np.random.default_rng(SEED)
    if len(held) < N_HELD_EN and remaining:
        held |= set(rng_h.choice(
            remaining, size=min(N_HELD_EN - len(held), len(remaining)), replace=False
        ).tolist())
    train_gens = [g for g in all_gens if g not in held]
    sp_train_en, sp_ood_en = sp[sp.gen.isin(train_gens)], sp[sp.gen.isin(held)]


    # ---- ASVspoof2019 LA train protocol ------------------------------------
    asv, flac = _parse_asv_eval()
    asv_bona, asv_spoof = asv[asv.key == "bonafide"], asv[asv.key == "spoof"]

    def asv_row(r, lab, split, group):
        return _row(flac / f"{r.utt}.flac", lab, split, "ASVspoof2019_LA",
                    f"asv_{r.spk}", f"asv_{r.utt}",
                    "none" if lab == "bonafide" else r.sys, "en",
                    "ASVspoof_EULA", "yes", group,
                    held_out=("yes" if split == "eval_ood" else "no"))

    # ---- In-the-Wild: SPEAKER-disjoint train vs eval_ood -------------------
    meta = pd.read_csv(ITW_ROOT / "meta.csv")
    meta.columns = [c.strip().lower() for c in meta.columns]
    meta["lab"] = meta["label"].map(
        lambda v: "bonafide" if str(v).strip().lower() in
        ("bona-fide", "bonafide", "real", "genuine") else "spoof"
    )
    spk_all = sorted(meta.speaker.astype(str).unique().tolist())
    rng_s = np.random.default_rng(SEED + 7)
    ood_spk = set(rng_s.choice(spk_all, size=max(1, len(spk_all) // 5), replace=False).tolist())
    itw_train = meta[~meta.speaker.astype(str).isin(ood_spk)]
    itw_ood = meta[meta.speaker.astype(str).isin(ood_spk)]

    def itw_row(r, split, group):
        return _row(ITW_ROOT / r.file, r.lab, split, "In-the-Wild",
                    _safe(r.speaker), f"itw_{_safe(Path(r.file).stem)}",
                    "none" if r.lab == "bonafide" else "itw-unknown", "en",
                    "CC-BY-4.0", "research", group,
                    held_out=("yes" if split == "eval_ood" else "no"))

    # ===== TRAIN (strict 1:1, sized by the bonafide ceiling) ================
    itw_tr_bo = itw_train[itw_train.lab == "bonafide"]
    itw_tr_sp = itw_train[itw_train.lab == "spoof"]
    en_train = len(asv_bona) + len(itw_tr_bo)

    for r in asv_bona.itertuples(index=False):
        out.append(asv_row(r, "bonafide", "train", "en_train"))
    out += [itw_row(r, "train", "en_train") for r in itw_tr_bo.itertuples(index=False)]

    # spoof mix: 25% ASV A01-A06, 25% ITW real-world spoof, mlaad-en fills the
    # rest. Each source is clamped to what it actually has and the remainder
    # rolls into mlaad-en (118 gens, ~118k clips), so 1:1 holds exactly.
    n_asv = min(en_train // 4, len(asv_spoof))
    n_itw = min(en_train // 4, len(itw_tr_sp))
    n_ml = en_train - n_asv - n_itw
    for r in stratified_sample(asv_spoof, n_asv, "sys", rng).itertuples(index=False):
        out.append(asv_row(r, "spoof", "train", "en_train"))
    out += [itw_row(r, "train", "en_train") for r in
            itw_tr_sp.sample(n_itw, random_state=SEED).itertuples(index=False)]
    out += _mlaad_rows(stratified_sample(sp_train_en, n_ml, "gen", rng),
                       "en", "train", "en_train")
    print(f"  [english] train {en_train}/class  "
          f"(bona: ASV {len(asv_bona)} + ITW {len(itw_tr_bo)} | "
          f"spoof: mlaad {n_ml} + ASV {n_asv} + ITW {n_itw})")

    # ===== EVAL_OOD (reported SEPARATELY; per-group pairing) ================
    itw_ood_bo = itw_ood[itw_ood.lab == "bonafide"].sample(frac=1.0, random_state=SEED)
    itw_ood_sp = itw_ood[itw_ood.lab == "spoof"]
    half = len(itw_ood_bo) // 2
    ref_bo, itw_bo = itw_ood_bo.iloc[:half], itw_ood_bo.iloc[half:]   # disjoint slices

    # (a) unseen mlaad-en generators vs pristine held-out real speech
    n_ood_ml = min(EN_OOD_SPOOF, len(sp_ood_en))
    out += _mlaad_rows(stratified_sample(sp_ood_en, n_ood_ml, "gen", rng),
                       "en", "eval_ood", "ood_en_mlaad")
    out += [itw_row(r, "eval_ood", "ood_en_mlaad") for r in ref_bo.itertuples(index=False)]

    # (b) real-world In-the-Wild, held-out speakers, both classes 1:1
    k_itw = min(len(itw_bo), len(itw_ood_sp))
    out += [itw_row(r, "eval_ood", "ood_itw") for r in itw_bo.head(k_itw).itertuples(index=False)]
    out += [itw_row(r, "eval_ood", "ood_itw") for r in
            itw_ood_sp.sample(k_itw, random_state=SEED).itertuples(index=False)]
    print(f"  [english] eval_ood: {len(held)} unseen gens x {n_ood_ml} clips vs "
          f"{len(ref_bo)} pristine real | ood_itw {k_itw}/class "
          f"({len(ood_spk)} held-out speakers)")

    return pd.DataFrame(out)


# ===========================================================================
# Pool 4 - Kannada RAW: Kathbath-kn bonafide  vs  mlaad/fake/kn spoof (2 gens)
#   Spoof-limited. Bonafide eval hold-out is by SPEAKER, spoof by row (it has
#   no speaker), so no clip and no voice crosses train -> eval.
# ===========================================================================
def pool_indic_raw(rng) -> pd.DataFrame:
    out = []
    for dirname, iso in INDIC_KEEP.items():
        bona = _list_audio(KATHBATH_DIR / dirname)
        sp = _mlaad_spoof_df(iso)
        if not bona or len(sp) == 0:
            print(f"  [indic-{iso}] SKIP (bona={len(bona)} spoof={len(sp)})")
            continue
        bo_df = pd.DataFrame({"wav": bona})
        bo_df["spk"] = bo_df.wav.map(lambda p: (Path(p).stem.split("-") + ["0"])[1])
        k = min(len(bona), len(sp))              # uncapped
        ke = max(1, k // KN_EVAL_DIV)
        sp_pick = stratified_sample(sp, k, "gen", rng)
        bo_pick = stratified_sample(bo_df, k, "spk", rng)
        # bonafide holds out whole SPEAKERS, so the eval size lands on a speaker
        # boundary; the spoof side (no speaker) is then cut to the SAME count so
        # both splits stay exactly 1:1.
        spk_order = list(bo_pick.groupby("spk").size().sort_values().index)  # smallest first
        ev_spk, acc = set(), 0
        for s in spk_order:
            if acc >= ke:
                break
            ev_spk.add(s)
            acc += int((bo_pick.spk == s).sum())
        bo_ev = bo_pick[bo_pick.spk.isin(ev_spk)]
        bo_tr = bo_pick[~bo_pick.spk.isin(ev_spk)]
        sp_ev, sp_tr = sp_pick.iloc[:len(bo_ev)], sp_pick.iloc[len(bo_ev):]
        out += _mlaad_rows(sp_tr, iso, "train", f"{iso}_train")
        out += _mlaad_rows(sp_ev, iso, "eval", f"indomain_{iso}")
        for sub, split, grp in [(bo_tr, "train", f"{iso}_train"),
                                (bo_ev, "eval", f"indomain_{iso}")]:
            for r in sub.itertuples(index=False):
                out.append(_row(
                    r.wav, "bonafide", split, "Kathbath", f"kb{iso}_{_safe(r.spk)}",
                    f"kb{iso}_{_safe(Path(r.wav).stem)}", "none", iso,
                    "CC-BY-4.0", "yes", grp))
        print(f"  [indic-{iso}] bona={len(bona)} spoof={len(sp)} -> {k}/class; "
              f"eval speakers={len(ev_spk)} (train bona {len(bo_tr)}, eval bona {len(bo_ev)})")
    return pd.DataFrame(out)


LANG_NAME = {"hi": "Hindi", "en": "English", "kn": "Kannada"}

PURPOSE = {
    "dev":            ("early-stop (best-EER)",             "Hindi official dev"),
    "eval":           ("in-domain sanity (held-out spk)",   "Hindi + Kannada, held-out speakers"),
    "ood_en_mlaad":   ("unseen generators",                 "held-out MLAAD-en gens vs pristine ITW real"),
    "ood_itw":        ("real-world held-out",               "disjoint In-the-Wild speakers, both classes"),
}


def _srcs(df, label):
    """Human-readable source list for one (language, label) cell."""
    sub = df[df.label == label]
    if not len(sub):
        return "-"
    parts = []
    for s in sorted(sub.source_dataset.unique()):
        g = sub[sub.source_dataset == s]
        ngen = g[g.generator_id != "none"].generator_id.nunique()
        parts.append(f"{s} ({ngen} gens)" if ngen > 1 else s)
    return " + ".join(parts)


def _table(rows, headers):
    w = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join("-" * x for x in w)
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print(line)
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def report(df: pd.DataFrame):
    tr = df[df.split == "train"]
    nb, ns = (tr.label == "bonafide").sum(), (tr.label == "spoof").sum()
    ngen = tr[tr.label == "spoof"].generator_id.nunique()
    print(f"\n=== TRAIN - {nb:,} bonafide / {ns:,} spoof - {ngen} spoof generators - 1:1 per language ===")
    rows = []
    for lang in ["hi", "en", "kn"]:
        g = tr[tr.language == lang]
        if not len(g):
            continue
        rows.append([LANG_NAME.get(lang, lang), _srcs(g, "bonafide"), _srcs(g, "spoof"),
                     f"{(g.label == 'bonafide').sum():,}", f"{(g.label == 'spoof').sum():,}"])
    rows.append(["TRAIN total", "", "", f"{nb:,}", f"{ns:,}"])
    _table(rows, ["Language", "Bonafide source", "Spoof source", "Bona", "Spoof"])

    print("\n=== DEV / EVAL / EVAL_OOD ===")
    rows = []
    for key in ["dev", "eval", "ood_en_mlaad", "ood_itw"]:
        g = df[df.group == key] if key.startswith("ood_") else df[df.split == key]
        if not len(g):
            continue
        purpose, comp = PURPOSE[key]
        name = key if key in ("dev", "eval") else f"eval_ood - {key}"
        rows.append([name, purpose, comp,
                     f"{(g.label == 'bonafide').sum():,}", f"{(g.label == 'spoof').sum():,}"])
    _table(rows, ["Split", "Purpose", "Composition", "Bona", "Spoof"])
    print(f"\nbase clips total: {len(df):,}  (VAD slicing multiplies this at stage 2)")


def sanity(df: pd.DataFrame):
    """Fail loudly on the two invariants that have burned this project."""
    problems = []
    tr = df[df.split == "train"]
    for lang, g in tr.groupby("language"):
        nb, ns = (g.label == "bonafide").sum(), (g.label == "spoof").sum()
        if nb != ns:
            problems.append(f"TRAIN {lang} is not 1:1 ({nb} bona vs {ns} spoof) -- language becomes a label proxy")
    # no speaker may appear in more than one split
    for spk, g in df[df.label == "bonafide"].groupby("speaker_id"):
        if g.split.nunique() > 1:
            problems.append(f"bonafide speaker {spk} spans splits {sorted(g.split.unique())}")
            break
    # eval_ood generators must be unseen in train
    tr_gens = set(tr[tr.label == "spoof"].generator_id)
    ood_gens = set(df[(df.group == "ood_en_mlaad")].generator_id) - {"none"}
    leaked = tr_gens & ood_gens
    if leaked:
        problems.append(f"eval_ood generators leaked into TRAIN: {sorted(leaked)[:5]}")
    if problems:
        print("\n[FAIL] " + "\n[FAIL] ".join(problems))
        return False
    print("\n[OK] 1:1 per language, speaker-disjoint splits, eval_ood generators unseen in train")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-hindi", default=str(MANIFEST_DIR / "hybrid_max_hindi_processed.csv"))
    ap.add_argument("--out-new", default=str(MANIFEST_DIR / "hybrid_max_new_raw.csv"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split without writing any manifest")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    print(f"Building HYBRID-MAX manifest (seed={SEED}); mlaad/fake langs = en, hi, kn\n")

    hi_p = pool_hindi_processed(rng); print("[hindi-processed] rows:", len(hi_p))
    hi_r = pool_hindi_raw(rng);       print("[hindi-raw/GV]    rows:", len(hi_r))
    en_r = pool_english_raw(rng);     print("[english-raw]     rows:", len(en_r))
    in_r = pool_indic_raw(rng);       print("[indic-raw]       rows:", len(in_r))

    hi_p = hi_p[FINAL_COLS]
    new = pd.concat([hi_r, en_r, in_r], ignore_index=True)[FINAL_COLS]
    combined = pd.concat([hi_p, new], ignore_index=True)

    report(combined)
    ok = sanity(combined)

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return 0 if ok else 1
    Path(args.out_hindi).parent.mkdir(parents=True, exist_ok=True)
    hi_p.to_csv(args.out_hindi, index=False, encoding="utf-8-sig")
    new.to_csv(args.out_new, index=False, encoding="utf-8-sig")
    print(f"\n[OK] wrote {args.out_hindi}  ({len(hi_p):,} Hindi passthrough rows)")
    print(f"[OK] wrote {args.out_new}  ({len(new):,} raw rows to preprocess)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
