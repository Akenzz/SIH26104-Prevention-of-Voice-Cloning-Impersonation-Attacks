// Browser port of the relay-service MockRVC "Talk as Teammate 3" spoof chain.
//
// The Flutter two-phone demo warps the caller's live voice inside the Python
// relay (`relay-service/converters/mock.py` -> MockRVCConverter): a three-stage
// DSP recipe that makes the speaker sound like a different person AND injects the
// exact neural-vocoder fingerprints the anti-spoof experts key on. This module
// reproduces the same recipe entirely in the browser so the frontend Live
// Monitor can drive the detector without the relay, two phones, or a GPU.
//
// The chain (order matches mock.py PitchFormantFallbackConverter.convert):
//   1. pitch shift        -- change of fundamental frequency
//   2. formant filter     -- peaking biquad, warps vocal-tract resonance
//   3. artifact injection -- tanh harmonic saturation + comb coloration
//   4. RMS match          -- restore the original loudness
//
// Fidelity note: stages 2, 3 and 4 are ported 1:1 from mock.py (same formulas,
// same coefficients; we pass the real 48 kHz worklet sampleRate so every cutoff
// lands on the same Hz). Stage 1 in mock.py is an offline STFT phase vocoder,
// which cannot run sample-synchronously on 128-frame worklet blocks; here it is
// a real-time granular (two-tap crossfade) pitch shifter. It produces the same
// perceptual + spectral pitch change; the artifact + formant stages -- which are
// what actually trip WavLM / LFCC-LCNN -- stay identical.

// Speaker profiles, copied verbatim from MockRVCConverter.PROFILES.
export const SPOOF_PROFILES = {
  clone_female: { pitch: 4.0, formant: 1.18, intensity: 0.14 },
  clone_male: { pitch: -3.5, formant: 0.88, intensity: 0.14 },
  clone_deep: { pitch: -6.0, formant: 0.8, intensity: 0.16 },
  clone_child: { pitch: 6.5, formant: 1.25, intensity: 0.15 },
  robotic: { pitch: 2.0, formant: 1.1, intensity: 0.25 },
};

// Human-facing labels for the profile picker.
export const SPOOF_PROFILE_LABELS = {
  clone_female: 'Clone · Female',
  clone_male: 'Clone · Male',
  clone_deep: 'Clone · Deep',
  clone_child: 'Clone · Child',
  robotic: 'Robotic',
};

export const DEFAULT_SPOOF_PROFILE = 'clone_female';

// The AudioWorklet processor source. It is compiled to a data: URI and loaded
// with `audioCtx.audioWorklet.addModule(...)`. Everything the worklet needs is
// inlined so the string is self-contained (and can be eval'd in a Node parity
// test against relay-service/converters/mock.py). Profiles are interpolated so
// the JS object above stays the single source of truth.
export function buildSpoofWorkletCode() {
  return `
const PROFILES = ${JSON.stringify(SPOOF_PROFILES)};
const DEFAULT_PROFILE = ${JSON.stringify(DEFAULT_SPOOF_PROFILE)};

// --- Stage 2: formant peaking biquad (mock.py apply_formant_filter) ----------
// Streaming Direct-Form-I biquad. Coeffs are the RBJ peaking-EQ cookbook, the
// same ones mock.py builds; recomputed only when the profile changes. With zero
// initial state a single block matches scipy.signal.lfilter exactly.
class FormantFilter {
  constructor(sampleRate) {
    this.sr = sampleRate;
    this.b0 = 1; this.b1 = 0; this.b2 = 0; this.a1 = 0; this.a2 = 0;
    this.x1 = 0; this.x2 = 0; this.y1 = 0; this.y2 = 0;
    this.bypass = true;
  }
  setRatio(formantRatio) {
    if (Math.abs(formantRatio - 1.0) < 0.02) { this.bypass = true; return; }
    this.bypass = false;
    let fc = 2800.0 * formantRatio;
    fc = Math.min(Math.max(fc, 800.0), 6500.0);
    const q = 2.5;
    const gainDb = formantRatio > 1.0 ? 6.0 : -4.0;
    const w0 = (2.0 * Math.PI * fc) / this.sr;
    const alpha = Math.sin(w0) / (2.0 * q);
    const A = Math.pow(10.0, gainDb / 40.0);
    const b0 = 1.0 + alpha * A;
    const b1 = -2.0 * Math.cos(w0);
    const b2 = 1.0 - alpha * A;
    const a0 = 1.0 + alpha / A;
    const a1 = -2.0 * Math.cos(w0);
    const a2 = 1.0 - alpha / A;
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
  }
  reset() { this.x1 = this.x2 = this.y1 = this.y2 = 0; }
  process(buf) {
    if (this.bypass) return buf;
    for (let i = 0; i < buf.length; i++) {
      const x = buf[i];
      const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2
              - this.a1 * this.y1 - this.a2 * this.y2;
      this.x2 = this.x1; this.x1 = x;
      this.y2 = this.y1; this.y1 = y;
      buf[i] = y;
    }
    return buf;
  }
}

// --- Stage 3: neural-vocoder artifact injection (mock.py) --------------------
// blended = x + intensity*(0.6*(tanh(1.8x) - x) + 0.4*comb(x)) where
// comb(x)[i] = x[i-delay] (zero before the stream starts). The last 'delay'
// input samples are carried across blocks so there is no per-block seam.
class ArtifactInjector {
  constructor(sampleRate) {
    this.intensity = 0;
    this.delay = Math.max(2, Math.floor(sampleRate / 3500.0));
    this.hist = new Float32Array(this.delay);
  }
  setIntensity(v) { this.intensity = v; }
  reset() { this.hist.fill(0); }
  process(buf) {
    const it = this.intensity;
    if (it <= 0.0) return buf;
    const d = this.delay;
    const hist = this.hist;           // length === d
    const inCopy = buf.slice();       // inputs to this stage
    for (let i = 0; i < buf.length; i++) {
      const x = inCopy[i];
      const harmonic = Math.tanh(1.8 * x) - x;
      const combSrc = i >= d ? inCopy[i - d] : hist[i];
      buf[i] = x + it * (0.6 * harmonic + 0.4 * combSrc);
    }
    if (inCopy.length >= d) {
      hist.set(inCopy.subarray(inCopy.length - d));
    } else {
      hist.copyWithin(0, inCopy.length);
      hist.set(inCopy, d - inCopy.length);
    }
    return buf;
  }
}

// --- Stage 1: real-time granular pitch shifter -------------------------------
// Two read taps a half-grain apart, triangular-windowed so they sum to unity;
// the tap that wraps is always at zero gain, hiding the discontinuity. Read
// speed = 2^(semitones/12); the tap phase walks at (readSpeed - 1) per sample.
class PitchShifter {
  constructor() {
    this.L = 4096;              // circular buffer length (> grain)
    this.buf = new Float32Array(this.L);
    this.w = 0;                 // write index
    this.grain = 1024;          // crossfade grain length (~21ms @48k warmup)
    this.frac = 0;              // tap phase in [0, grain)
    this.ratio = 1;
    this.bypass = true;
  }
  setSemitones(semitones) {
    if (Math.abs(semitones) < 0.05) { this.bypass = true; this.ratio = 1; return; }
    this.bypass = false;
    this.ratio = Math.pow(2.0, semitones / 12.0);
  }
  reset() { this.buf.fill(0); this.w = 0; this.frac = 0; }
  _readInterp(pos) {
    let p = pos % this.L;
    if (p < 0) p += this.L;
    const i0 = Math.floor(p);
    const i1 = (i0 + 1) % this.L;
    const a = p - i0;
    return this.buf[i0] * (1 - a) + this.buf[i1] * a;
  }
  process(inBuf) {
    if (this.bypass) return inBuf;
    const out = new Float32Array(inBuf.length);
    const grain = this.grain;
    const half = grain / 2;
    const step = this.ratio - 1.0;
    for (let i = 0; i < inBuf.length; i++) {
      this.buf[this.w] = inBuf[i];
      const p1 = this.frac;
      let p2 = this.frac + half;
      if (p2 >= grain) p2 -= grain;
      const g1 = 1.0 - Math.abs((2.0 * p1) / grain - 1.0);
      const g2 = 1.0 - Math.abs((2.0 * p2) / grain - 1.0);
      out[i] = g1 * this._readInterp(this.w - p1) + g2 * this._readInterp(this.w - p2);
      this.frac += step;
      if (this.frac >= grain) this.frac -= grain;
      else if (this.frac < 0) this.frac += grain;
      this.w = (this.w + 1) % this.L;
    }
    return out;
  }
}

class SpoofCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.spoof = false;
    this.pitch = new PitchShifter();
    this.formant = new FormantFilter(sampleRate);
    this.artifacts = new ArtifactInjector(sampleRate);
    this._applyProfile(DEFAULT_PROFILE);
    this.port.onmessage = (e) => {
      const d = e.data || {};
      if (d.type === 'config') {
        if (typeof d.spoofEnabled === 'boolean') this.spoof = d.spoofEnabled;
        if (d.profile && PROFILES[d.profile]) this._applyProfile(d.profile);
      } else if (d.type === 'reset') {
        this.pitch.reset(); this.formant.reset(); this.artifacts.reset();
      }
    };
  }

  _applyProfile(name) {
    const p = PROFILES[name] || PROFILES[DEFAULT_PROFILE];
    this.pitch.setSemitones(p.pitch);
    this.formant.setRatio(p.formant);
    this.artifacts.setIntensity(p.intensity);
  }

  _convert(input) {
    let inSumSq = 0;
    for (let i = 0; i < input.length; i++) inSumSq += input[i] * input[i];

    let out = this.pitch.process(input.slice()); // fresh buffer, input untouched
    out = this.formant.process(out);              // in place
    out = this.artifacts.process(out);            // in place

    let outSumSq = 0;
    for (let i = 0; i < out.length; i++) outSumSq += out[i] * out[i];
    const inRms = Math.sqrt(inSumSq / input.length);
    const outRms = Math.sqrt(outSumSq / out.length);
    if (outRms > 1e-6 && inRms > 1e-6) {
      // Cap the boost so a near-silent granular-pitch warmup block (right after
      // spoof-on / reset) fades in instead of exploding into a burst. Normal
      // matching is ~1x; only the first ~grain samples ever need the clamp.
      const g = Math.min(inRms / outRms, 4.0);
      for (let i = 0; i < out.length; i++) out[i] *= g;
    }
    return out;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      if (channelData) {
        // postMessage without a transfer list copies, so the audio thread can
        // safely reuse channelData; slice() on the clean path keeps that true.
        this.port.postMessage(this.spoof ? this._convert(channelData) : channelData.slice());
      }
    }
    return true;
  }
}

registerProcessor('spoof-capture-processor', SpoofCaptureProcessor);
`;
}
