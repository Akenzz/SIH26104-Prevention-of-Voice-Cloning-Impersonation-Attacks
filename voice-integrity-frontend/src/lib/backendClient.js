// WebSocket transport to the task C realtime backend.
//
// Handshake: send JSON start -> receive {type:"ready"} -> stream binary
// pcm_f32le frames -> receive {type:"score"|"error"} messages.

export function wsUrlFromBase(base) {
  // base "" -> same-origin /ws (dev proxy). Otherwise accept http(s)/ws(s) URLs.
  if (!base) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws`;
  }
  let u = base.trim().replace(/\/+$/, "");
  u = u.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  if (!/^wss?:/.test(u)) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    u = `${proto}//${u}`;
  }
  return u.endsWith("/ws") ? u : `${u}/ws`;
}

export function healthUrlFromBase(base) {
  if (!base) return "/health";
  const u = base.trim().replace(/\/+$/, "").replace(/^ws:/, "http:").replace(/^wss:/, "https:");
  return `${u}/health`;
}

export async function fetchHealth(base, { signal } = {}) {
  const res = await fetch(healthUrlFromBase(base), { signal });
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}

export class BackendSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.ready = false;
    this._queue = [];
    this.handlers = {};
  }

  on(event, fn) {
    this.handlers[event] = fn;
    return this;
  }
  _emit(event, ...args) {
    if (this.handlers[event]) this.handlers[event](...args);
  }

  connect({ sampleRate, encoding = "pcm_f32le", channels = 1 }) {
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => {
      this._emit("open");
      this.ws.send(
        JSON.stringify({ type: "start", sample_rate: sampleRate, encoding, channels })
      );
    };
    this.ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "ready") {
        this.ready = true;
        this._flush();
        this._emit("ready", msg);
        return;
      }
      this._emit("message", msg);
    };
    this.ws.onerror = (e) => this._emit("error", e);
    this.ws.onclose = (e) => {
      this.ready = false;
      this._emit("close", e);
    };
    return this;
  }

  _flush() {
    if (!this.ready || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    for (const buf of this._queue) this.ws.send(buf);
    this._queue.length = 0;
  }

  // Accepts an ArrayBuffer of little-endian float32 samples.
  sendFrame(arrayBuffer) {
    if (!this.ws) return;
    if (!this.ready || this.ws.readyState !== WebSocket.OPEN) {
      // buffer a little audio captured during the handshake, then drop the excess
      if (this._queue.length < 64) this._queue.push(arrayBuffer);
      return;
    }
    this.ws.send(arrayBuffer);
  }

  close() {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
    }
    this.ws = null;
    this.ready = false;
    this._queue.length = 0;
  }
}
