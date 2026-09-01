import os
import sys
import time
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download

REPO = "mueller91/MLAAD"
DEST = r"E:\DatasetSIH\mlaad"
AUDIO_EXT = (".wav", ".flac")

PER_GENERATOR = {
    "en": 50,
    "hi": 150,
    "kn": 300,
    "ml": 300,
    "mr": 300,
    "ta": 300,
}
DEFAULT_CAP = 150
LANGS = set(PER_GENERATOR.keys())


def banner(text):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}", flush=True)


def fmt_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


banner("STEP 1/4  Fetching repo file list from Hugging Face")
print(f"  Repo: {REPO}", flush=True)
t0 = time.time()
api = HfApi()
all_files = api.list_repo_files(REPO, repo_type="dataset")
print(f"  Retrieved {len(all_files):,} paths in {time.time()-t0:.1f}s", flush=True)


banner("STEP 2/4  Grouping audio by language + generator")
buckets = defaultdict(list)
for f in all_files:
    parts = f.split("/")
    if len(parts) >= 4 and parts[0] == "fake" and parts[1] in LANGS:
        if f.lower().endswith(AUDIO_EXT):
            buckets[(parts[1], parts[2])].append(f)

meta_files = [
    f for f in all_files
    if f.endswith("meta.csv")
    and len(f.split("/")) >= 2
    and f.split("/")[1] in LANGS
]

gens_per_lang = defaultdict(int)
for (lang, _gen) in buckets:
    gens_per_lang[lang] += 1

print(f"  meta.csv files found: {len(meta_files)}", flush=True)
print(f"  Generators per language:", flush=True)
for lang in sorted(LANGS):
    print(f"     {lang:<3} : {gens_per_lang.get(lang, 0):>4} generators", flush=True)


banner("STEP 3/4  Building capped download list")
to_download = list(meta_files)
per_lang_counts = defaultdict(int)
for (lang, gen), files in sorted(buckets.items()):
    files.sort()
    cap = PER_GENERATOR.get(lang, DEFAULT_CAP)
    chosen = files[:cap]
    to_download.extend(chosen)
    per_lang_counts[lang] += len(chosen)

print(f"  {'LANG':<6}{'GENERATORS':>12}{'CLIPS':>10}", flush=True)
print(f"  {'-'*28}", flush=True)
for lang in sorted(per_lang_counts):
    print(f"  {lang:<6}{gens_per_lang.get(lang,0):>12}{per_lang_counts[lang]:>10}", flush=True)
print(f"  {'-'*28}", flush=True)
print(f"  {'TOTAL':<6}{len(buckets):>12}{sum(per_lang_counts.values()):>10}", flush=True)
print(f"\n  meta.csv: {len(meta_files)}   |   Grand total files: {len(to_download):,}", flush=True)


banner(f"STEP 4/4  Downloading {len(to_download):,} files (sequential, resumable)")
total = len(to_download)
start = time.time()
skipped = 0
downloaded = 0
failed = 0

for i, f in enumerate(to_download, 1):
    existed = os.path.exists(os.path.join(DEST, f.replace("/", os.sep)))
    short = f if len(f) <= 60 else "..." + f[-57:]
    try:
        hf_hub_download(
            REPO, filename=f, repo_type="dataset",
            local_dir=DEST, local_dir_use_symlinks=False,
        )
        if existed:
            skipped += 1
            tag = "SKIP"
        else:
            downloaded += 1
            tag = "GET "
    except Exception as e:
        failed += 1
        tag = "FAIL"
        print(f"\n  [{i}/{total}] FAIL {short}\n         {e}", flush=True)

    elapsed = time.time() - start
    rate = i / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate if rate > 0 else 0
    pct = i / total * 100
    bar_len = 30
    filled = int(bar_len * i / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(
        f"\r  [{bar}] {pct:5.1f}%  {i}/{total}  "
        f"got={downloaded} skip={skipped} fail={failed}  "
        f"ETA {fmt_eta(eta)}   {tag} {short[-40:]:<40}"
    )
    sys.stdout.flush()

print(flush=True)
banner("DONE")
print(f"  Downloaded : {downloaded}", flush=True)
print(f"  Skipped    : {skipped}  (already cached)", flush=True)
print(f"  Failed     : {failed}", flush=True)
print(f"  Elapsed    : {fmt_eta(time.time()-start)}", flush=True)
print(f"  Files under: {DEST}", flush=True)