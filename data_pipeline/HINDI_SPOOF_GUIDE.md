# Hindi Spoof Dataset — Complete Build Guide

**Goal:** turn ~86k bonafide AI4Bharat **Kathbath Hindi** clips into a small, balanced, leakage-free
**bonafide + spoof** dataset for the V2 detector, using **multiple generators** so the model learns
real spoof cues instead of one TTS's fingerprint.

This guide is the single source of truth for the whole pipeline. Steps 1–2 are **done** (artifacts on
disk). Steps 3–8 are what's ahead, with exact commands and the Colab code.

---

## 0. Design at a glance (why it's built this way)

| Decision | What we do | Why |
|---|---|---|
| **Matched pairs** | every spoof clip also exists as its own bonafide row (same speaker, same sentence) | detector can't cheat on speaker / content / gender |
| **Speaker-disjoint splits** | a speaker's clips live in exactly ONE of train/dev/eval | no utterance or speaker leakage across splits |
| **Multiple generators** | IndicF5 + XTTS-v2 in train; +RVC in eval | generalization — not "stuck to one TTS" |
| **Disjoint in train** | each train/dev clip spoofed by **one** generator (~50/50 IndicF5/XTTS *within each speaker*) | cheaper, keeps matched pairs, no speaker↔generator confound |
| **Overlap in eval** | each eval clip spoofed by **every** generator | per-generator EER is apples-to-apples on identical content |
| **RVC = held-out** | RVC appears **only** in eval | measures generalization to an **unseen** generator (README task H) |
| **Real-only eval slice** | 3k bonafide-only eval clips, no spoof | measures false-positive rate on unseen real speakers |
| **No forced 50/50 gender** | keep natural balance | matched pairs already neutralize gender as a shortcut |
| ~~IndicSynth~~ | **dropped** | not matched-content, extra plumbing; RVC already gives a clean held-out generator |

**Generators**
- **IndicF5** (ai4bharat/IndicF5) — flow-matching TTS; clones from *reference audio + its transcript*. Native Indic.
- **XTTS-v2** (coqui/XTTS-v2) — autoregressive TTS + diffusion; clones from a reference clip; `language="hi"`.
- **RVC** — voice *conversion* (keeps source words/prosody, swaps timbre). Held-out. Ties to the live-demo clone (task G). **Consenting teammate voice only — never a public figure.**

---

## 1. ✅ Recover transcripts — DONE

`fetch_kathbath.py` saved only the wavs and dropped the `text` column. IndicF5/XTTS need the sentence to
regenerate matched content, so we read it back from the cached parquet shards.

```bash
python data_pipeline/recover_kathbath_transcripts.py
```

**Result:** `manifests/kathbath_hindi_transcripts.csv` — 94,903 transcripts, **86,302 / 86,302 wavs matched (100%)**.

---

## 2. ✅ Plan the jobs — DONE

Decide the speaker split, sample a balanced budget, and assign generators. **Generates no audio.**

```bash
python data_pipeline/plan_hindi_spoof_jobs.py
```

**Artifacts (verified against real data — 0 speaker overlap, 71/71 train speakers gender-balanced ±1):**

| File | What |
|---|---|
| `manifests/hindi_spoof_jobs.csv` | 8,800 jobs — the Colab work order |
| `manifests/hindi_bonafide_selected.csv` | 10,600 bonafide rows — the REAL half of the dataset |
| `manifests/hindi_refs_to_upload.txt` | 7,600 ref wavs to send to Colab |
| `reports/hindi_spoof_plan_summary.json` | counts, speaker split, budgets |

Jobs by split × generator: **train** 3032 f5 / 2968 xtts · **dev** 506 / 494 · **eval** 600 / 600 / 600 (rvc eval-only).

> Tune scale with flags, e.g. `--train-budget 8000 --eval-budget 800`. Re-run is deterministic (seed 1337).
> If you change budgets, **re-run every downstream step**.

---

## 3. Stage & upload the reference clips

Copy the 7,600 ref wavs into a clean folder, zip, and upload to Google Drive.

```bash
python data_pipeline/plan_hindi_spoof_jobs.py --stage-refs
```

This writes `E:\DatasetSIH\hindi_spoof\refs\<split>\<utt>.wav`. Zip that `refs/` folder plus the two CSVs:

```bash
cd /e/DatasetSIH/hindi_spoof && \
  cp "/d/SIH/SIH26104-Prevention-of-Voice-Cloning-Impersonation-Attacks/data_pipeline/manifests/hindi_spoof_jobs.csv" . && \
  tar -a -c -f hindi_refs.zip refs hindi_spoof_jobs.csv
```

Upload `hindi_refs.zip` to Drive (e.g. `MyDrive/sih/hindi_refs.zip`). ~7,600 clips ≈ **0.7–1.0 GB**.

---

## 4. Generate on Colab (free GPU)

**One notebook per generator.** Each cell mounts Drive, reads `hindi_spoof_jobs.csv`, filters to its
generator, and writes to each job's exact `out_relpath` (`spoof/<split>/<gen>/<utt>.wav`). All three are
**resumable** — a clip already on disk is skipped, so you can stop/restart a session freely.

> ⚠️ TTS APIs drift. If a call signature has changed, check the model card on Hugging Face and adjust the
> one generate line — the harness (loop, resume, paths) stays the same.

### Common setup cell (run first in every notebook)

```python
from google.colab import drive; drive.mount('/content/drive')
import os, zipfile, pandas as pd
from pathlib import Path

ZIP = '/content/drive/MyDrive/sih/hindi_refs.zip'
WORK = Path('/content/work'); WORK.mkdir(exist_ok=True)
if not (WORK/'refs').exists():
    with zipfile.ZipFile(ZIP) as z: z.extractall(WORK)

OUT = Path('/content/drive/MyDrive/sih/out')   # generated audio lands here
jobs = pd.read_csv(WORK/'hindi_spoof_jobs.csv', encoding='utf-8-sig', dtype={'speaker_id':str})

# map utt_id -> ref wav (layout-independent: scan whatever the zip contained)
refmap = {p.stem: p for p in (WORK/'refs').rglob('*.wav')}
print('jobs:', len(jobs), '| refs on disk:', len(refmap))

def ref_for(utt):        # ref clip for a job
    return refmap.get(utt)
def dst_for(relpath):    # out path on Drive, mkdir parents
    d = OUT/relpath; d.parent.mkdir(parents=True, exist_ok=True); return d
```

### 4a. XTTS-v2  (`target_generator == "xtts"`)

```python
!pip -q install TTS
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

todo = jobs[jobs.target_generator == "xtts"]
for i, j in enumerate(todo.itertuples(index=False), 1):
    out = dst_for(j.out_relpath)
    if out.exists(): continue
    ref = ref_for(j.utt_id)
    if ref is None: print("no ref:", j.utt_id); continue
    try:
        tts.tts_to_file(text=j.text, speaker_wav=str(ref), language="hi", file_path=str(out))
    except Exception as e:
        print("FAIL", j.utt_id, e)
    if i % 100 == 0: print(f"{i}/{len(todo)}")
print("xtts done")
```

### 4b. IndicF5  (`target_generator == "indicf5"`)

Matched content = feed the **same** sentence as both the reference transcript and the target text.

```python
!pip -q install transformers soundfile
import soundfile as sf, numpy as np
from transformers import AutoModel
model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to("cuda")

todo = jobs[jobs.target_generator == "indicf5"]
for i, j in enumerate(todo.itertuples(index=False), 1):
    out = dst_for(j.out_relpath)
    if out.exists(): continue
    ref = ref_for(j.utt_id)
    if ref is None: print("no ref:", j.utt_id); continue
    try:
        wav = model(j.text, ref_audio_path=str(ref), ref_text=j.text)   # verify arg names on the model card
        wav = np.asarray(wav, dtype=np.float32)
        if wav.max() > 1.5: wav = wav / 32768.0        # some builds return int16-scaled
        sf.write(str(out), wav, 24000)                  # IndicF5 is 24 kHz; parity pass resamples to 16k
    except Exception as e:
        print("FAIL", j.utt_id, e)
    if i % 100 == 0: print(f"{i}/{len(todo)}")
print("indicf5 done")
```

### 4c. RVC — held-out (`target_generator == "rvc"`, eval only, 600 clips)

RVC needs a **trained target-voice model** (`.pth` + `.index`) from a **consenting teammate**. This is the
same clone you demo live (task G). Train it once from ~5–10 min of their speech (RVC WebUI or `rvc-python`),
put `teammate.pth`/`teammate.index` on Drive, then batch-convert the 600 eval source clips:

```python
!pip -q install rvc-python
from rvc_python.infer import RVCInference
rvc = RVCInference(model_path="/content/drive/MyDrive/sih/rvc/teammate.pth",
                   index_path="/content/drive/MyDrive/sih/rvc/teammate.index", device="cuda:0")

todo = jobs[jobs.target_generator == "rvc"]
for i, j in enumerate(todo.itertuples(index=False), 1):
    out = dst_for(j.out_relpath)
    if out.exists(): continue
    ref = ref_for(j.utt_id)              # RVC converts the source clip's timbre -> teammate
    if ref is None: print("no ref:", j.utt_id); continue
    try:
        rvc.infer_file(input_path=str(ref), output_path=str(out))
    except Exception as e:
        print("FAIL", j.utt_id, e)
    if i % 50 == 0: print(f"{i}/{len(todo)}")
print("rvc done")
```

**Rough GPU time (T4):** XTTS ~3.5k clips ≈ 2–4 h · IndicF5 ~3.5k ≈ 1.5–3 h · RVC 600 ≈ 20–40 min.
XTTS and IndicF5 can run in **parallel Colab sessions**.

---

## 5. Download generated audio

Bring the Drive `out/` folder down so it mirrors each job's `out_relpath` under one root:

```
E:\DatasetSIH\hindi_spoof\spoof\<split>\<generator>\<utt>.wav
```

i.e. copy `MyDrive/sih/out/spoof/...` → `E:\DatasetSIH\hindi_spoof\spoof\...` (keep the `spoof/` layout).

---

## 6. Build the merged manifest

Turn every job whose audio now exists into a spoof row and merge with the bonafide half. Resumable — run it
after each generator finishes to get an interim manifest; missing audio is skipped and counted.

```bash
python data_pipeline/build_hindi_manifest.py
```

→ `manifests/hindi_v2_full.csv` (12-col schema + `held_out`/`role`/`text`; `held_out=yes` on RVC rows).
Prints label×split and generator×split tables + how many jobs still lack audio.

---

## 7. Preprocessing parity + shortcut canary  ← the methodology that makes the numbers trustworthy

Real recordings and TTS output differ in leading silence, loudness, and file fingerprints. If you don't
flatten those **identically for both classes**, the detector learns "less silence ⇒ fake" instead of a real
cue. So: run the same pass over every clip, then prove a trivial classifier can't separate them.

**7a. Canary BEFORE (expect a warning — raw TTS has tells):**
```bash
python data_pipeline/canary_silence.py --manifest data_pipeline/manifests/hindi_v2_full.csv
```

**7b. Uniform pass — 16 kHz mono → silence-trim → loudness-normalize, into a `processed/` mirror:**
```bash
pip install librosa pyloudnorm
python data_pipeline/preprocess_parity.py \
  --manifest    data_pipeline/manifests/hindi_v2_full.csv \
  --out-audio   E:/DatasetSIH/hindi_spoof/processed \
  --out-manifest data_pipeline/manifests/hindi_v2_processed.csv
```

**7c. Canary AFTER (must collapse to ~chance, AUC ≈ 0.5):**
```bash
python data_pipeline/canary_silence.py --manifest data_pipeline/manifests/hindi_v2_processed.csv
```
If AUC stays > 0.65 after parity, **stop** — fix preprocessing before trusting any EER.

---

## 8. Leakage check + fold into V2

**8a. Prove zero leakage** (speaker/utterance disjoint; held-out generator absent from train):
```bash
python data_pipeline/check_leakage.py \
  --manifest data_pipeline/manifests/hindi_v2_processed.csv \
  --held-out-gens rvc-consented
```
Expect `[PASS] Zero data leakage`.

**8b. Merge into the V2 training manifest** and retrain. The Hindi rows plug into the existing V2 build
(`build_v2_manifests.py`) alongside ASVspoof19 / MLAAD; the Hindi-specific expert trains on
`hindi_v2_processed.csv`. Re-fit the calibrator whenever the expert or fusion changes — the shipped Platt
calibrator is in-domain only.

---

## Pipeline map

```
recover_kathbath_transcripts.py ─▶ kathbath_hindi_transcripts.csv
plan_hindi_spoof_jobs.py ──┬─▶ hindi_spoof_jobs.csv ──▶ [Colab ×3] ──▶ E:\...\hindi_spoof\spoof\...
                           ├─▶ hindi_bonafide_selected.csv ─┐
                           └─▶ (--stage-refs) refs\ ─▶ zip ─┘
build_hindi_manifest.py ───▶ hindi_v2_full.csv
preprocess_parity.py ──────▶ hindi_v2_processed.csv  (+ processed\ audio)
canary_silence.py ─────────▶ shortcut check (before & after)
check_leakage.py ──────────▶ [PASS] ──▶ fold into build_v2_manifests.py / retrain
```

---

## Honesty & consent (README Claims Ledger)

- These spoofs are **synthetic training data**, not proof the detector "prevents voice-cloning fraud" or
  "proves identity." Report EER/accuracy per generator, and separately for the **held-out RVC** as the real
  generalization signal — don't fold held-out numbers into headline in-domain metrics.
- RVC / any voice clone: **consenting team members only, never public figures.**
- Note licenses in the manifest: Kathbath **CC-BY-4.0**; XTTS **CPML (non-commercial)**; IndicF5 research-only;
  RVC output = consented teammate. Fine for a research/hackathon dataset; don't relabel as own numbers or
  ship commercially.
```
