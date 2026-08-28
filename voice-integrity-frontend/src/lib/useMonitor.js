import { useCallback, useEffect, useRef, useState } from "react";
import { AudioEngine, floatFrameToBytes } from "./audio.js";
import { BackendSocket, fetchHealth, wsUrlFromBase } from "./backendClient.js";
import { Simulator } from "./simulator.js";
import { narrate, verdictLine } from "./narrator.js";

const SERIES_CAP = 160; // ~80 s of history at a 0.5 s hop
const REASON_CAP = 60;
let LINE_ID = 0;

// One source-agnostic controller for the console. It drives either the local
// Simulator or a live backend WebSocket (mic or file) and exposes the same
// derived state to every panel.
export function useMonitor() {
  const [source, setSource] = useState("sim"); // "sim" | "live"
  const [scenario, setScenario] = useState("clone");
  const [inputKind, setInputKind] = useState("mic"); // "mic" | "file"
  const [backendBase, setBackendBase] = useState("");

  const [status, setStatus] = useState("idle"); // idle|connecting|running|stopped|error
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

  const pushReason = useCallback((line) => {
    if (!line) return;
    setReasoning((r) => {
      const next = [...r, { id: ++LINE_ID, ...line }];
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
      setCurrent(msg);

      setStats((s) => {
        const seq = msg.sequence_number;
        let seqOk = s.seqOk;
        if (expectSeqRef.current != null && seq !== expectSeqRef.current) seqOk = false;
        expectSeqRef.current = seq + 1;
        return {
          count: s.count + 1,
          dropped: s.dropped + (msg.dropped_frames ? 1 : 0),
          maxLatency: Math.max(s.maxLatency, msg.latency_ms || 0),
          lastSeq: seq,
          seqOk,
        };
      });

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

  const teardown = useCallback(() => {
    if (simRef.current) simRef.current.stop();
    simRef.current = null;
    if (socketRef.current) socketRef.current.close();
    socketRef.current = null;
    if (engineRef.current) engineRef.current.stop();
    startedAtRef.current = null;
  }, []);

  const stop = useCallback(() => {
    const last = lastMsgRef.current;
    teardown();
    setStatus("stopped");
    const v = verdictLine(last);
    if (v) pushReason(v);
  }, [teardown, pushReason]);

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
          setStatus((s) => (s === "error" ? s : "stopped"));
        });
      socket.connect({ sampleRate, encoding: "pcm_f32le", channels: 1 });

      const onFrame = (frame) => {
        const sock = socketRef.current;
        if (sock) sock.sendFrame(floatFrameToBytes(frame));
      };

      startedAtRef.current = performance.now();
      try {
        if (file) {
          await engine.startFile(file, { onFrame, onEnded: () => stop() });
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
    [backendBase, handleMessage, resetRun, stop, teardown]
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

  const running = status === "running" || status === "connecting";
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
