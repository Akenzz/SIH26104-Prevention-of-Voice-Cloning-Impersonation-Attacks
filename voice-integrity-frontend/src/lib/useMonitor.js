import { useCallback, useEffect, useRef, useState } from "react";
import { AudioEngine, floatFrameToBytes } from "./audio.js";
import { BackendSocket, fetchHealth, wsUrlFromBase } from "./backendClient.js";
import { Simulator } from "./simulator.js";
import { narrate, verdictLine } from "./narrator.js";

const SERIES_CAP = 160; // ~80 s of history at a 0.5 s hop
const REASON_CAP = 60;
let LINE_ID = 0;

// A file is pushed to the backend far faster than the detector scores it — a
// 6 s clip uploads in well under a second, while WavLM on CPU needs seconds per
// 4 s window. So when the file ends we keep the socket open and drain: close
// only after the backend has gone quiet for DRAIN_IDLE_MS, with a hard cap so a
// wedged backend can't leave the console spinning forever.
const DRAIN_IDLE_MS = 12000;
const DRAIN_MAX_MS = 180000;

// One source-agnostic controller for the console. It drives either the local
// Simulator or a live backend WebSocket (mic or file) and exposes the same
// derived state to every panel.
export function useMonitor() {
  const [source, setSource] = useState("sim"); // "sim" | "live"
  const [scenario, setScenario] = useState("clone");
  const [inputKind, setInputKind] = useState("mic"); // "mic" | "file"
  const [backendBase, setBackendBase] = useState("");

  const [status, setStatus] = useState("idle"); // idle|connecting|running|draining|stopped|error
  const [error, setError] = useState(null);
  const [current, setCurrent] = useState(null);
  const [series, setSeries] = useState([]);
  const [reasoning, setReasoning] = useState([]);
  const [stats, setStats] = useState({ count: 0, dropped: 0, maxLatency: 0, lastSeq: null, seqOk: true });
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  const engineRef = useRef(null);
  const socketRef = useRef(null);
  const simRef = useRef(null);
  const prevMsgRef = useRef(null);
  const lastMsgRef = useRef(null);
  const expectSeqRef = useRef(null);
  const startedAtRef = useRef(null);
  const phaseRef = useRef(0);
  const drainRef = useRef(null); // { idle: timeoutId, cap: timeoutId, bump: fn } while draining

  const pushReason = useCallback((line) => {
    if (!line) return;
    const entry = { id: ++LINE_ID, ...line }; // id assigned outside the updater: keys must be stable
    setReasoning((r) => {
      const next = [...r, entry];
      return next.length > REASON_CAP ? next.slice(next.length - REASON_CAP) : next;
    });
  }, []);

  const handleMessage = useCallback(
    (msg) => {
      if (msg.type === "ready") {
        setStatus("running");
        return;
      }
      // score or error frame
      lastMsgRef.current = msg;
      if (drainRef.current) drainRef.current.bump(); // backend is still answering
      setCurrent(msg);

      // Sequence bookkeeping happens out here on purpose: React re-runs state
      // updaters in dev (StrictMode) to catch impure ones, so advancing a ref
      // inside setStats would skip a number and report a phantom gap.
      const seq = msg.sequence_number;
      const gap = expectSeqRef.current != null && seq !== expectSeqRef.current;
      expectSeqRef.current = seq + 1;

      setStats((s) => ({
        count: s.count + 1,
        dropped: s.dropped + (msg.dropped_frames ? 1 : 0),
        maxLatency: Math.max(s.maxLatency, msg.latency_ms || 0),
        lastSeq: seq,
        seqOk: s.seqOk && !gap,
      }));

      setSeries((arr) => {
        const point = {
          idx: (arr.length ? arr[arr.length - 1].idx : 0) + 1,
          p: msg.smoothed_probability,
          state: msg.risk_state,
          window: msg.window_index,
        };
        const next = [...arr, point];
        return next.length > SERIES_CAP ? next.slice(next.length - SERIES_CAP) : next;
      });

      const line = narrate(prevMsgRef.current, msg);
      pushReason(line);
      prevMsgRef.current = msg;
    },
    [pushReason]
  );

  const resetRun = useCallback(() => {
    prevMsgRef.current = null;
    lastMsgRef.current = null;
    expectSeqRef.current = null;
    setCurrent(null);
    setSeries([]);
    setReasoning([]);
    setStats({ count: 0, dropped: 0, maxLatency: 0, lastSeq: null, seqOk: true });
    setError(null);
  }, []);

  const clearDrain = useCallback(() => {
    if (!drainRef.current) return;
    clearTimeout(drainRef.current.idle);
    clearTimeout(drainRef.current.cap);
    drainRef.current = null;
  }, []);

  const teardown = useCallback(() => {
    clearDrain();
    if (simRef.current) simRef.current.stop();
    simRef.current = null;
    if (socketRef.current) socketRef.current.close();
    socketRef.current = null;
    if (engineRef.current) engineRef.current.stop();
    startedAtRef.current = null;
  }, [clearDrain]);

  const stop = useCallback(() => {
    const last = lastMsgRef.current;
    teardown();
    setStatus("stopped");
    const v = verdictLine(last);
    if (v) pushReason(v);
  }, [teardown, pushReason]);

  // File playback finished: stop pushing audio but keep the socket open so
  // windows still in the detector come back. Each arriving score re-arms the
  // idle timer via drainRef.bump().
  const beginDrain = useCallback(() => {
    if (engineRef.current) engineRef.current.stop();
    if (!socketRef.current) {
      stop();
      return;
    }
    setStatus("draining");
    const finish = () => {
      clearDrain();
      stop();
    };
    drainRef.current = {
      idle: setTimeout(finish, DRAIN_IDLE_MS),
      cap: setTimeout(finish, DRAIN_MAX_MS),
      bump: () => {
        const d = drainRef.current;
        if (!d) return;
        clearTimeout(d.idle);
        d.idle = setTimeout(finish, DRAIN_IDLE_MS);
      },
    };
  }, [clearDrain, stop]);

  const startSim = useCallback(() => {
    resetRun();
    const sim = new Simulator({ scenario });
    simRef.current = sim;
    startedAtRef.current = performance.now();
    setStatus("running");
    sim.start(handleMessage);
  }, [scenario, handleMessage, resetRun]);

  const startLive = useCallback(
    async (file) => {
      teardown(); // drop any socket/engine/timer left over from a previous run
      resetRun();
      setStatus("connecting");
      const engine = new AudioEngine();
      engineRef.current = engine;
      let sampleRate;
      try {
        ({ sampleRate } = await engine.prepare());
      } catch (err) {
        setError("Could not open the audio system: " + err.message);
        setStatus("error");
        return;
      }

      const socket = new BackendSocket(wsUrlFromBase(backendBase));
      socketRef.current = socket;
      socket
        .on("ready", () => setStatus("running"))
        .on("message", handleMessage)
        .on("error", () => {
          setError("WebSocket error. Is the backend running on the configured address?");
          setStatus("error");
        })
        .on("close", () => {
          setStatus((s) => (s === "error" || s === "draining" ? s : "stopped"));
        });
      socket.connect({ sampleRate, encoding: "pcm_f32le", channels: 1 });

      const onFrame = (frame) => {
        const sock = socketRef.current;
        if (sock) sock.sendFrame(floatFrameToBytes(frame));
      };

      startedAtRef.current = performance.now();
      try {
        if (file) {
          await engine.startFile(file, { onFrame, onEnded: beginDrain });
        } else {
          await engine.startMic({ onFrame });
        }
      } catch (err) {
        if (err && err.name === "NotAllowedError") {
          setError("Microphone permission denied. Allow mic access or use a file / simulation.");
        } else {
          setError("Audio capture failed: " + (err?.message || err));
        }
        setStatus("error");
        teardown();
      }
    },
    [backendBase, handleMessage, resetRun, beginDrain, teardown]
  );

  const start = useCallback(
    (file) => {
      if (source === "sim") startSim();
      else startLive(file);
    },
    [source, startSim, startLive]
  );

  const refreshHealth = useCallback(async () => {
    setHealthError(null);
    try {
      const h = await fetchHealth(backendBase);
      setHealth(h);
      return h;
    } catch (err) {
      setHealth(null);
      setHealthError(err.message || "unreachable");
      return null;
    }
  }, [backendBase]);

  // Elapsed-time ticker while running.
  useEffect(() => {
    if (status !== "running") return;
    const id = setInterval(() => {
      if (startedAtRef.current) setElapsedSec((performance.now() - startedAtRef.current) / 1000);
    }, 250);
    return () => clearInterval(id);
  }, [status]);

  // Stop everything on unmount.
  useEffect(() => () => teardown(), [teardown]);

  // Waveform provider for the level meter: real analyser when live, else a
  // synthetic trace whose amplitude tracks the current risk so sim still reads
  // as "live signal".
  const readWaveform = useCallback((out) => {
    const engine = engineRef.current;
    if (engine && engine.active) return engine.readWaveform(out);
    // synthesize
    const cur = lastMsgRef.current;
    const p = cur?.smoothed_probability ?? 0.1;
    const unavailable = cur?.risk_state === "unavailable";
    phaseRef.current += 0.28;
    const amp = unavailable ? 0.03 : 0.18 + 0.5 * p;
    let sum = 0;
    for (let i = 0; i < out.length; i++) {
      const t = i / out.length;
      const v =
        amp *
        (Math.sin(phaseRef.current + t * 22) * 0.6 +
          Math.sin(phaseRef.current * 0.7 + t * 55) * 0.3 +
          Math.sin(phaseRef.current * 1.9 + t * 8) * 0.1);
      out[i] = v;
      sum += v * v;
    }
    return Math.sqrt(sum / out.length);
  }, []);

  const running = status === "running" || status === "connecting" || status === "draining";
  const audioActive = source === "live";

  return {
    source, setSource,
    scenario, setScenario,
    inputKind, setInputKind,
    backendBase, setBackendBase,
    status, running, error,
    current, series, reasoning, stats,
    health, healthError, refreshHealth,
    elapsedSec, audioActive,
    start, stop, readWaveform,
  };
}
