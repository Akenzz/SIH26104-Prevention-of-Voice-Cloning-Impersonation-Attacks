"""A/B: does front-end normalization shrink the channel gap on local clips?

For each clip: POST the raw file, then POST a normalized copy
(DC removal -> high-pass 70 Hz -> RMS normalize -> 16 kHz mono),
and compare mc_v3 raw logits + calibrated result.
"""
import io, json, sys, wave, tempfile, statistics as st
from pathlib import Path
import numpy as np
import requests
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd

SRC = Path(r"E:\DatasetSIH\Local_test")
FILES = {"sarosh_omni": "spoof", "sarosh_spoof": "spoof",
         "sarosh_styletts": "spoof", "sudhanva": "bonafide"}
URL = "http://127.0.0.1:8000/predict-file"
TARGET_SR = 16000
TARGET_RMS = 0.06  # ~ -24 dBFS

def read_wav(p):
    with wave.open(str(p), "rb") as w:
        ch, sr, wd, n = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0 if wd == 2 else \
        np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr

def normalize(x, sr):
    x = x - np.mean(x)                          # DC removal
    sos = butter(4, 70.0, btype="high", fs=sr, output="sos")
    x = sosfilt(sos, x).astype(np.float32)      # high-pass 70 Hz
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR); x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    rms = float(np.sqrt(np.mean(x ** 2))) or 1.0
    x = x * (TARGET_RMS / rms)                  # RMS normalize
    return np.clip(x, -1.0, 1.0).astype(np.float32), TARGET_SR

def write_wav(x, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()

def post(data, name):
    r = requests.post(URL, files={"file": (name, data, "audio/wav")})
    r.raise_for_status(); d = r.json(); s = d["summary"]; ws = d["windows"]
    v = [w["raw_per_expert_scores"].get("mc_v3") for w in ws
         if w["raw_per_expert_scores"].get("mc_v3") is not None]
    return s["overall_risk_state"], s["final_smoothed_probability"] or 0, (st.mean(v) if v else 0), (min(v) if v else 0), (max(v) if v else 0)

print(f"{'file':16} {'truth':9} | raw: state prob  mc_v3(mean/min/max) | norm: state prob  mc_v3(mean/min/max)")
for name, truth in FILES.items():
    p = SRC / f"{name}.wav"
    x, sr = read_wav(p)
    r_state, r_p, r_m, r_lo, r_hi = post(p.read_bytes(), f"{name}.wav")
    nx, nsr = normalize(x, sr)
    n_state, n_p, n_m, n_lo, n_hi = post(write_wav(nx, nsr), f"{name}_norm.wav")
    print(f"{name:16} {truth:9} | {r_state:11} {r_p:.2f}  {r_m:+.2f}/{r_lo:+.2f}/{r_hi:+.2f} "
          f"| {n_state:11} {n_p:.2f}  {n_m:+.2f}/{n_lo:+.2f}/{n_hi:+.2f}")
