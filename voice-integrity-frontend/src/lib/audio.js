// Audio capture + playback for the monitor.
//
// Two jobs:
//  1. Produce mono Float32 PCM frames to stream to the backend (task C accepts
//     encoding "pcm_f32le" at any declared sample_rate and resamples to 16 kHz).
//  2. Expose an AnalyserNode so the UI can draw a live level meter / waveform.
//
// Capture uses an AudioWorklet (off the main thread; survives long sessions),
// falling back to a ScriptProcessorNode where AudioWorklet is unavailable.

const WORKLET_SRC = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opt = (options && options.processorOptions) || {};
    this.frame = opt.frameSamples || 2048;
    this.acc = new Float32Array(this.frame);
    this.n = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.acc[this.n++] = ch[i];
      if (this.n === this.frame) {
        this.port.postMessage(this.acc.slice(0));
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.node = null; // worklet or script processor
    this.analyser = null;
    this.sink = null; // muted gain -> destination, keeps the graph pulling
    this.player = null; // AudioBufferSourceNode for file playback
    this.mode = null; // "mic" | "file"
    this._stopStreaming = false;
  }

  get sampleRate() {
    return this.ctx ? this.ctx.sampleRate : TARGET_FALLBACK_RATE;
  }

  get active() {
    return this.mode != null;
  }

  // Fill a Float32Array with the current time-domain waveform (-1..1).
  readWaveform(out) {
    if (!this.analyser) {
      out.fill(0);
      return 0;
    }
    this.analyser.getFloatTimeDomainData(out);
    let sum = 0;
    for (let i = 0; i < out.length; i++) sum += out[i] * out[i];
    return Math.sqrt(sum / out.length); // rms
  }

  // Create the AudioContext up front so the caller can read sampleRate and
  // open the WebSocket handshake before any capture begins.
  async prepare() {
    await this._ensureContext();
    return { sampleRate: this.ctx.sampleRate };
  }

  async _ensureContext() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
    }
    if (this.ctx.state === "suspended") await this.ctx.resume();
    if (!this.analyser) {
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.6;
    }
  }

  async _makeCaptureNode(onFrame) {
    const frameSamples = Math.max(
      256,
      Math.round(this.ctx.sampleRate * 0.064) // ~64 ms frames
    );
    if (this.ctx.audioWorklet) {
      try {
        const blob = new Blob([WORKLET_SRC], { type: "application/javascript" });
        const url = URL.createObjectURL(blob);
        await this.ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        const node = new AudioWorkletNode(this.ctx, "capture-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          processorOptions: { frameSamples },
        });
        node.port.onmessage = (e) => onFrame(e.data);
        return node;
      } catch (err) {
        // fall through to ScriptProcessor
        console.warn("AudioWorklet unavailable, using ScriptProcessor", err);
      }
    }
    const node = this.ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => {
      onFrame(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    return node;
  }

  async startMic({ onFrame, onError } = {}) {
    await this._ensureContext();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    } catch (err) {
      if (onError) onError(err);
      throw err;
    }
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = await this._makeCaptureNode(onFrame || (() => {}));
    // source -> analyser (meter) and source -> capture -> muted sink (keeps
    // ScriptProcessor firing; mic is never routed audibly to avoid feedback).
    this.source.connect(this.analyser);
    this.source.connect(this.node);
    this.sink = this.ctx.createGain();
    this.sink.gain.value = 0;
    this.node.connect(this.sink);
    this.sink.connect(this.ctx.destination);
    this.mode = "mic";
    return { sampleRate: this.ctx.sampleRate, channels: 1 };
  }

  // Decode a file and stream it through the same frame path, paced in real time
  // so the timeline animates. Audio is played audibly so the room hears the call.
  async startFile(file, { onFrame, onLevel, onEnded, onError } = {}) {
    await this._ensureContext();
    let audioBuffer;
    try {
      const bytes = await file.arrayBuffer();
      audioBuffer = await this.ctx.decodeAudioData(bytes);
    } catch (err) {
      if (onError) onError(err);
      throw err;
    }
    const rate = audioBuffer.sampleRate;
    const data = audioBuffer.getChannelData(0);

    // Audible playback through the analyser so the meter is live.
    this.player = this.ctx.createBufferSource();
    this.player.buffer = audioBuffer;
    this.player.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this.mode = "file";
    this._stopStreaming = false;
    this.player.start();

    // Frame streaming, paced to wall clock.
    const frameSamples = Math.max(256, Math.round(rate * 0.064));
    const frameMs = (frameSamples / rate) * 1000;
    let offset = 0;
    const t0 = performance.now();
    let framesSent = 0;

    const pump = () => {
      if (this._stopStreaming) return;
      const dueBy = performance.now() - t0 + frameMs;
      // send every frame whose scheduled time has arrived
      while (offset < data.length && (framesSent * frameMs) <= dueBy) {
        const end = Math.min(offset + frameSamples, data.length);
        const chunk = data.slice(offset, end);
        if (onFrame) onFrame(chunk);
        offset = end;
        framesSent += 1;
      }
      if (onLevel) {
        const wf = new Float32Array(1024);
        const rms = this.readWaveform(wf);
        onLevel(rms);
      }
      if (offset < data.length) {
        this._pumpTimer = setTimeout(pump, frameMs);
      } else if (onEnded) {
        onEnded();
      }
    };
    pump();
    return { sampleRate: rate, channels: 1, durationSec: audioBuffer.duration };
  }

  stop() {
    this._stopStreaming = true;
    if (this._pumpTimer) clearTimeout(this._pumpTimer);
    this._pumpTimer = null;
    try {
      if (this.player) this.player.stop();
    } catch {
      /* already stopped */
    }
    this.player = null;
    for (const n of [this.node, this.source, this.sink]) {
      try {
        if (n) n.disconnect();
      } catch {
        /* ignore */
      }
    }
    this.node = null;
    this.source = null;
    this.sink = null;
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.mode = null;
  }

  async close() {
    this.stop();
    if (this.ctx) {
      try {
        await this.ctx.close();
      } catch {
        /* ignore */
      }
      this.ctx = null;
      this.analyser = null;
    }
  }
}

const TARGET_FALLBACK_RATE = 48000;

// Encode a Float32 frame as little-endian bytes for pcm_f32le transport.
export function floatFrameToBytes(frame) {
  // frame is already Float32Array; its buffer is LE on all supported platforms.
  return frame.buffer.slice(
    frame.byteOffset,
    frame.byteOffset + frame.byteLength
  );
}
