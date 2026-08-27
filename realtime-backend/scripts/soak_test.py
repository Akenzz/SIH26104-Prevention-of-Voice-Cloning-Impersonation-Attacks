"""Loop audio through the in-process pipeline and watch RSS.

Default is a few minutes with DummyExpert. Pass --minutes 10 for the
Task C 10–15 minute bar. Memory must stay roughly flat.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import load_calibrator
from config import ARTIFACTS_DIR, Settings
from experts.loader import load_experts
from fusion import load_fusion
from pipeline import ConnectionState
from policy import load_policy


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=2.0)
    parser.add_argument("--log-every-sec", type=float, default=15.0)
    parser.add_argument(
        "--experts",
        default="dummy",
        help="comma list of experts to soak: dummy, lfcc, wavlm (real experts load from Hub)",
    )
    args = parser.parse_args()

    expert_names = [e.strip() for e in args.experts.split(",") if e.strip()]
    settings = Settings(experts=expert_names, fusion_mode="single")
    experts = load_experts(settings)
    print(f"soak experts={ {k: v.model_version for k, v in experts.items()} }")
    state = ConnectionState(
        settings=settings,
        experts=experts,
        fusion=load_fusion(ARTIFACTS_DIR / "fusion.json"),
        calibrator=load_calibrator(ARTIFACTS_DIR / "calibrator.json"),
        policy=load_policy(ARTIFACTS_DIR / "policy.json"),
    )
    state.apply_start(
        {"type": "start", "sample_rate": 16000, "encoding": "pcm_f32le", "channels": 1}
    )

    hop = settings.hop_samples
    rng = np.random.default_rng(11)
    loop = rng.normal(0, 0.1, hop * 8).astype(np.float32)
    pcm_chunks = [
        loop[i : i + hop].astype("<f4").tobytes() for i in range(0, loop.size, hop)
    ]

    deadline = time.time() + args.minutes * 60.0
    next_log = time.time()
    rss_samples: list[float] = []
    latencies: list[float] = []
    seq = 0
    idx = 0
    start_rss = rss_mb()
    print(f"soak start rss_mb={start_rss:.1f} minutes={args.minutes}")

    while time.time() < deadline:
        messages = state.ingest_binary(pcm_chunks[idx % len(pcm_chunks)])
        idx += 1
        for msg in messages:
            seq = msg["sequence_number"]
            if msg.get("latency_ms") is not None:
                latencies.append(float(msg["latency_ms"]))
            if msg["risk_state"] == "unavailable":
                raise SystemExit(f"unexpected unavailable during soak: {msg}")
        now = time.time()
        if now >= next_log:
            rss = rss_mb()
            rss_samples.append(rss)
            p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (
                max(latencies) if latencies else float("nan")
            )
            print(
                f"t={args.minutes * 60 - (deadline - now):.0f}s seq={seq} "
                f"rss_mb={rss:.1f} p95_latency_ms={p95:.2f} n={len(latencies)}"
            )
            next_log = now + args.log_every_sec

    end_rss = rss_mb()
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (
        max(latencies) if latencies else float("nan")
    )
    hop_ms = settings.hop_sec * 1000.0
    print(
        f"soak done seq={seq} start_rss={start_rss:.1f} end_rss={end_rss:.1f} "
        f"p95_latency_ms={p95:.2f} hop_ms={hop_ms:.0f}"
    )
    if seq < 1:
        raise SystemExit("no scores produced")
    if not np.isnan(end_rss) and not np.isnan(start_rss) and end_rss - start_rss > 150:
        raise SystemExit(f"RSS grew too much: {start_rss:.1f} -> {end_rss:.1f} MB")
    if latencies and p95 >= hop_ms:
        raise SystemExit(f"p95 latency {p95:.2f} ms is not below hop {hop_ms:.0f} ms")
    print("OK: memory bounded, sequence increased, p95 latency below hop")


if __name__ == "__main__":
    main()
