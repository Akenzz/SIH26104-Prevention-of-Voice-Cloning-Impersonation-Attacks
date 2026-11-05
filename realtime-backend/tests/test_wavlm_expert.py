"""
tests/test_wavlm_expert.py — Smoke tests for Expert 1 (WavLM adapter).

Skips cleanly when:
  - transformers is not installed, or
  - the checkpoint is missing (neither local nor Hub available).

Run just this test:
    cd realtime-backend
    pytest tests/test_wavlm_expert.py -v

Run the EER benchmark against the full ASVspoof test split:
    pytest tests/test_wavlm_expert.py -v -k eer --no-header
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("transformers", reason="transformers not installed")

from config import HUB_EXPERTS, Settings


def test_wavlm_uses_hub_when_local_checkpoint_is_stale(tmp_path, monkeypatch):
    """A stale local copy must not win over the newly uploaded Hub model."""
    from experts import wavlm as wavlm_module

    local_path = tmp_path / "best_model_v5.pt"
    local_path.write_bytes(b"old-weight-data")
    cache_dir = tmp_path / "cache"
    expected = cache_dir / HUB_EXPERTS["wavlm"]["local_name"]

    monkeypatch.setattr(wavlm_module, "_LOCAL_CHECKPOINT", local_path)
    monkeypatch.setattr(
        wavlm_module,
        "_remote_checkpoint_is_stale",
        lambda repo_id, filename, local: True,
    )

    seen = {}

    def fake_ensure_checkpoint(repo_id, filename, cache_dir, local_name=None):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        seen["cache_dir"] = cache_dir
        seen["local_name"] = local_name
        return expected

    monkeypatch.setattr(wavlm_module, "ensure_checkpoint", fake_ensure_checkpoint)

    resolved = wavlm_module._resolve_wavlm_checkpoint(cache_dir)

    assert resolved == expected
    assert seen["repo_id"] == HUB_EXPERTS["wavlm"]["repo_id"]
    assert seen["filename"] == HUB_EXPERTS["wavlm"]["filename"]


def _load_expert(device: str = "cpu"):
    """Load WavLMExpert, skip the test if unavailable."""
    from experts.wavlm import WavLMExpert

    settings = Settings()
    try:
        return WavLMExpert(cache_dir=settings.model_cache_dir, device=device)
    except Exception as exc:
        pytest.skip(f"WavLM expert unavailable in this environment: {exc}")


# ─── Basic contract tests ──────────────────────────────────────────────────────

def test_wavlm_expert_returns_correct_score_keys():
    expert = _load_expert()
    window = np.zeros(64_000, dtype=np.float32)
    score  = expert.score(window)
    assert set(score.keys()) == {"logit", "embedding", "model_version"}


def test_wavlm_expert_returns_finite_logit():
    expert = _load_expert()
    rng    = np.random.default_rng(42)
    window = rng.normal(0.0, 0.1, 64_000).astype(np.float32)
    score  = expert.score(window)
    assert isinstance(score["logit"], float)
    assert np.isfinite(score["logit"])


def test_wavlm_expert_returns_768_dim_embedding():
    expert = _load_expert()
    window = np.random.default_rng(7).normal(0.0, 0.1, 64_000).astype(np.float32)
    score  = expert.score(window)
    assert score["embedding"] is not None
    assert len(score["embedding"]) == 768
    assert all(np.isfinite(v) for v in score["embedding"])


def test_wavlm_expert_is_deterministic():
    """Frozen backbone + eval mode → identical logit on repeated calls."""
    expert = _load_expert()
    window = np.random.default_rng(1).normal(0.0, 0.1, 64_000).astype(np.float32)
    a = expert.score(window)["logit"]
    b = expert.score(window)["logit"]
    assert abs(a - b) < 1e-5, f"Non-deterministic: {a} vs {b}"


def test_wavlm_expert_handles_short_window():
    """32 000 samples (2 s) should be zero-padded without crashing."""
    expert = _load_expert()
    short  = np.random.default_rng(2).normal(0.0, 0.1, 32_000).astype(np.float32)
    assert np.isfinite(expert.score(short)["logit"])


def test_wavlm_expert_handles_long_window():
    """96 000 samples (6 s) should be cropped without crashing."""
    expert = _load_expert()
    long   = np.random.default_rng(3).normal(0.0, 0.1, 96_000).astype(np.float32)
    assert np.isfinite(expert.score(long)["logit"])


def test_wavlm_model_version_set():
    expert = _load_expert()
    assert "wavlm" in expert.model_version.lower()
    assert expert.model_version != "wavlm-unwired"


# ─── EER benchmark (marked slow — run explicitly with -k eer) ─────────────────

@pytest.mark.slow
def test_wavlm_eer_on_asvspoof_test_split():
    """
    Compute EER on a balanced 5 000-sample subset of the ASVspoof 2019 LA
    test split.  Requires:
        expert1/data/asvspoof_manifest.csv   (created by repath_manifest.py)
        expert1/checkpoints/best_model.pt    (created by train.py)

    Run with:
        pytest tests/test_wavlm_expert.py -v -k eer
    """
    import random
    import importlib
    from pathlib import Path
    import sys

    # Make wavlm-base-plus importable
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        pkg = importlib.import_module("wavlm-base-plus.dataset")
        SpeechDataset = pkg.SpeechDataset
        from sklearn.metrics import roc_curve
    except ImportError as e:
        pytest.skip(f"wavlm-base-plus or sklearn not importable: {e}")

    MANIFEST = Path(repo_root) / "wavlm-base-plus" / "data" / "asvspoof_manifest.csv"
    if not MANIFEST.exists():
        pytest.skip(f"Manifest not found: {MANIFEST}  — run repath_manifest.py first")

    try:
        ds = SpeechDataset(str(MANIFEST), split="test")
    except Exception as e:
        pytest.skip(f"Could not load test dataset: {e}")

    # Build a balanced 5 000-sample subset
    MAX_SAMPLES = 5_000
    random.seed(42)
    bonafide_idx = [i for i in range(len(ds)) if ds.data.iloc[i]["label"] == "bonafide"]
    spoof_idx    = [i for i in range(len(ds)) if ds.data.iloc[i]["label"] == "spoof"]
    half         = MAX_SAMPLES // 2
    chosen       = (
        random.sample(bonafide_idx, min(half, len(bonafide_idx)))
        + random.sample(spoof_idx,  min(half, len(spoof_idx)))
    )
    random.shuffle(chosen)

    expert  = _load_expert()
    labels  = []
    logits  = []

    print(f"\n  Scoring {len(chosen)} samples from ASVspoof 2019 LA test split ...")
    for i, idx in enumerate(chosen, 1):
        waveform, label = ds[idx]
        score = expert.score(waveform.numpy())
        labels.append(int(label.item()))
        logits.append(score["logit"])
        if i % 500 == 0:
            print(f"  {i}/{len(chosen)} scored ...")

    labels_np = np.array(labels)
    logits_np = np.array(logits)

    # EER
    fpr, tpr, _ = roc_curve(labels_np, logits_np, pos_label=1)
    fnr         = 1.0 - tpr
    eer_idx     = np.argmin(np.abs(fpr - fnr))
    eer         = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    # Accuracy
    preds    = (logits_np > 0).astype(int)
    accuracy = (preds == labels_np).mean()

    n_bonafide = (labels_np == 0).sum()
    n_spoof    = (labels_np == 1).sum()

    print(f"\n  ══════════════════════════════════════")
    print(f"  WavLM Expert — EER Benchmark")
    print(f"  ══════════════════════════════════════")
    print(f"  Evaluated on : {len(labels):,} samples  (bonafide={n_bonafide}, spoof={n_spoof})")
    print(f"  Accuracy     : {accuracy:.1%}")
    print(f"  EER          : {eer * 100:.2f}%")
    print(f"  Checkpoint   : {expert.checkpoint_path}")
    print(f"  ══════════════════════════════════════\n")

    # Sanity guard: untrained random head → EER ≈ 50 %; wired model should do better.
    assert eer < 0.50, f"EER {eer:.2%} is no better than random — model may not be wired correctly"
