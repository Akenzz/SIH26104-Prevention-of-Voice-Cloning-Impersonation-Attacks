import { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, Activity, AlertTriangle, CheckCircle, HelpCircle, Zap, ZapOff } from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge, RiskBadge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { narrate, verdictLine } from '../lib/narrator';
import { narrateRemote } from '../lib/narrateRemote';
import ReasoningLog from '../components/panels/ReasoningLog';
import { buildSpoofWorkletCode, SPOOF_PROFILE_LABELS, DEFAULT_SPOOF_PROFILE } from '../lib/spoofDsp';

// The capture AudioWorklet doubles as the "Talk as Teammate 3 (Spoof)" engine:
// clean pass-through when spoof is off, and the MockRVC pitch/formant/artifact
// warp (ported from relay-service) applied in-thread when spoof is on. This is
// the same recipe the Flutter two-phone demo runs, so the browser can drive the
// detector into "high" without the relay, a second phone, or a GPU.
const workletCode = buildSpoofWorkletCode();

// Session recording. The capture worklet already hands us the exact Float32
// frames that go to the detector (clean, or spoof-warped when "Talk as
// Teammate 3" is on). We keep those frames in memory for the length of ONE
// monitoring session and, on stop, mux them into a 16-bit PCM WAV Blob so the
// operator can play back precisely what the model scored. Nothing touches disk;
// the Blob is revoked the moment the next session starts.
function concatFloat32(chunks) {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function encodeWavFromFloat32(samples, sampleRate) {
  const dataSize = samples.length * 2; // 16-bit mono
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);            // PCM fmt chunk size
  view.setUint16(20, 1, true);             // format = PCM
  view.setUint16(22, 1, true);             // channels = mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (sr * blockAlign)
  view.setUint16(32, 2, true);             // block align (mono * 16-bit)
  view.setUint16(34, 16, true);            // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: 'audio/wav' });
}

export default function LiveMonitor() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  
  const [scores, setScores] = useState([]);
  const [currentState, setCurrentState] = useState(null);
  const [currentProb, setCurrentProb] = useState(0);
  const [latestData, setLatestData] = useState(null);
  const [reasoning, setReasoning] = useState([]);

  // "Talk as Teammate 3 (Spoof)" state. Refs shadow the state so the worklet
  // message handlers and the ws.onopen config send always read the live value
  // without waiting for a re-render.
  const [spoofEnabled, setSpoofEnabled] = useState(false);
  const [spoofProfile, setSpoofProfile] = useState(DEFAULT_SPOOF_PROFILE);
  const spoofEnabledRef = useRef(false);
  const spoofProfileRef = useRef(DEFAULT_SPOOF_PROFILE);

  const wsRef = useRef(null);
  // Task F narration bookkeeping: the narrator is stateful (it only speaks on a
  // change), so we keep the previous and last score message; a monotonic id per
  // log line; and whether the backend LLM narration path is live this session.
  const prevMsgRef = useRef(null);
  const lastMsgRef = useRef(null);
  const lineIdRef = useRef(0);
  const narrationEnabledRef = useRef(false);
  const audioContextRef = useRef(null);
  const workletNodeRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Session playback. We buffer the exact frames the worklet hands us (clean or
  // spoof-warped — whatever the detector saw) and, on stop, mux them into an
  // in-memory WAV. Nothing is written to disk; the Blob URL is revoked when the
  // next session starts, so a recording only ever lives until the next Start.
  const recordedChunksRef = useRef([]);
  const recordSampleRateRef = useRef(48000);
  const sessionUsedSpoofRef = useRef(false);
  // Detector's peak call over the session — tracked separately from
  // sessionUsedSpoofRef so the playback card can show what the model actually
  // concluded (the verdict) next to whether the operator ran the voice-changer
  // (provenance). prob === -1 means no window was ever scored.
  const sessionPeakRef = useRef({ prob: -1, state: null });
  const recordingUrlRef = useRef(null);
  const mountedRef = useRef(true);
  const [recordingUrl, setRecordingUrl] = useState(null);
  const [recordingSpoofed, setRecordingSpoofed] = useState(false);
  const [recordingVerdict, setRecordingVerdict] = useState(null);

  // Swap in a new playback URL, revoking the previous Blob so nothing lingers.
  const setRecordingUrlSafe = useCallback((url) => {
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    recordingUrlRef.current = url;
    setRecordingUrl(url);
  }, []);

  const connectAndStart = async () => {
    try {
      setError(null);
      setStatus('connecting');

      // 1. Get Microphone — capture the RAWEST possible signal.
      // The detectors were trained on clean corpus PCM with no browser DSP in
      // the path. Chrome defaults echoCancellation / noiseSuppression /
      // autoGainControl to TRUE on a getUserMedia audio track, so leaving them
      // on inserts three processing stages the model never saw in training.
      // echoCancellation is the worst offender for the judge demo: its whole
      // job is to CANCEL audio coming from the speakers, so an AI/cloned clip
      // played near the mic gets partially suppressed — the very thing we want
      // the detector to hear. All three OFF keeps live capture closest to the
      // training distribution. (If a room is genuinely noisy you can flip these
      // back on to compare; play the source at a healthy volume so the window
      // never drops below the silence gate.)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      mediaStreamRef.current = stream;

      // 2. Setup AudioContext (we'll let it pick the default device rate, backend will resample)
      const audioCtx = new window.AudioContext();
      audioContextRef.current = audioCtx;
      const sampleRate = audioCtx.sampleRate;

      // 3. Connect WebSocket via proxy
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = async () => {
        setStatus('connected');
        setIsRecording(true);
        setScores([]);
        setReasoning([]);
        prevMsgRef.current = null;
        lastMsgRef.current = null;
        // Every session starts as your real voice; the operator opts into spoof.
        setSpoofEnabled(false);
        spoofEnabledRef.current = false;

        // Start a fresh recording and drop the previous session's playback, so a
        // clip only ever lives until the next Start (in-memory only, no disk).
        recordedChunksRef.current = [];
        recordSampleRateRef.current = sampleRate;
        sessionUsedSpoofRef.current = false;
        sessionPeakRef.current = { prob: -1, state: null };
        setRecordingSpoofed(false);
        setRecordingVerdict(null);
        setRecordingUrlSafe(null);

        // Ask the backend whether the optional Groq narration path is live. If
        // not (no GROQ_API_KEY), we silently use the local template narrator.
        try {
          const h = await fetch('/health').then((r) => r.json());
          narrationEnabledRef.current = !!h?.narration?.enabled;
        } catch {
          narrationEnabledRef.current = false;
        }

        // Send start message
        ws.send(JSON.stringify({
          type: 'start',
          sample_rate: sampleRate,
          encoding: 'pcm_f32le',
          channels: 1
        }));

        // 4. Setup AudioWorklet using a base64 Data URI (solves mobile/HTTPS Blob blocks)
        const base64Code = btoa(workletCode);
        const url = `data:application/javascript;base64,${base64Code}`;
        await audioCtx.audioWorklet.addModule(url);
        
        const source = audioCtx.createMediaStreamSource(stream);
        
        // Setup visual analyser
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        
        const drawWaveform = () => {
          if (!canvasRef.current) return;
          const canvas = canvasRef.current;
          const canvasCtx = canvas.getContext('2d');
          const width = canvas.width;
          const height = canvas.height;
          
          const bufferLength = analyser.frequencyBinCount;
          const dataArray = new Uint8Array(bufferLength);
          analyser.getByteTimeDomainData(dataArray);
          
          canvasCtx.clearRect(0, 0, width, height);
          canvasCtx.lineWidth = 2;
          canvasCtx.strokeStyle = 'rgba(16, 185, 129, 0.8)'; // emerald-500
          
          canvasCtx.beginPath();
          const sliceWidth = width * 1.0 / bufferLength;
          let x = 0;
          for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * height / 2;
            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);
            x += sliceWidth;
          }
          canvasCtx.lineTo(width, height / 2);
          canvasCtx.stroke();
          
          animationFrameRef.current = requestAnimationFrame(drawWaveform);
        };
        drawWaveform();

        const workletNode = new AudioWorkletNode(audioCtx, 'spoof-capture-processor');
        workletNodeRef.current = workletNode;
        // Prime the worklet with the current spoof mode + profile before any audio
        // flows, so a mid-session Start always begins in a known state.
        workletNode.port.postMessage({
          type: 'config',
          spoofEnabled: spoofEnabledRef.current,
          profile: spoofProfileRef.current,
        });

        workletNode.port.onmessage = (e) => {
          const f32Array = e.data;
          // Keep this frame for session playback: it is exactly what the detector
          // sees (clean, or spoof-warped when "Talk as Teammate 3" is on). The
          // worklet hands us a fresh buffer per block and ws.send copies rather
          // than transfers, so retaining the reference is safe.
          recordedChunksRef.current.push(f32Array);
          // Send raw binary float32 array (clean pass-through, or the spoof-warped
          // frame when "Talk as Teammate 3" is on — same bytes either way).
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(f32Array.buffer);
          }
        };

        source.connect(workletNode);
        // Some browsers stop scheduling an AudioWorkletNode that has no outgoing
        // connection, which would silence process() and starve both the detector
        // and the session recording. So we DO route the node to the destination —
        // but only through a muted (0-gain) node, so the live-warped output is
        // never played back through the speakers (which, with an open mic, would
        // feed back). Playback is muxed from the captured frames, not this path.
        const gain = audioCtx.createGain();
        gain.gain.value = 0;
        workletNode.connect(gain);
        gain.connect(audioCtx.destination);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'score') {
          // Per-expert probabilities live under data.scores[name].probability on
          // the /ws wire (there is no top-level data.expert_probabilities — that
          // key was always undefined, which is why the expert lines never drew).
          const expertProbs = Object.fromEntries(
            Object.entries(data.scores || {})
              // SSL (TakHemlata) is decommissioned — the lineup is two models
              // (WavLM + LFCC-LCNN). Drop any stray `ssl` score so the chart can
              // never render a third expert line.
              .filter(([k, v]) => k !== 'ssl' && typeof v?.probability === 'number')
              .map(([k, v]) => [k, v.probability])
          );
          const newScore = {
            time: (data.sequence_number * 0.5).toFixed(1), // Assuming 0.5s hop for display
            prob: data.smoothed_probability,
            lr_prob: data.lr_probability,
            state: data.risk_state,
            expert_probs: expertProbs,
          };
          
          setScores(prev => {
            const next = [...prev, newScore];
            if (next.length > 50) next.shift(); // Keep last 50 points
            return next;
          });
          
          setCurrentState(data.risk_state);
          setCurrentProb(data.smoothed_probability);
          setLatestData(data);

          // Remember the detector's peak call this session, so the playback card
          // can report what the model actually decided — independent of whether
          // the operator turned the voice-changer on. Only finite scores count
          // ("collecting"/quality-gated windows carry no probability).
          const peakP = data.smoothed_probability;
          if (typeof peakP === 'number' && Number.isFinite(peakP) && peakP > sessionPeakRef.current.prob) {
            sessionPeakRef.current = { prob: peakP, state: data.risk_state };
          }

          // Task F: narrate the change, if any. The local narrator is both the
          // event gate AND the offline fallback; the LLM only rephrases the same
          // grounded fields when enabled, so it adds no new hallucination surface.
          const prev = prevMsgRef.current;
          const local = narrate(prev, data);
          prevMsgRef.current = data;
          lastMsgRef.current = data;
          if (local) {
            const id = ++lineIdRef.current;
            if (narrationEnabledRef.current) {
              setReasoning((r) => [...r, { id, text: '', kind: local.kind, streaming: true }]);
              const fields = {
                window_index: data.window_index,
                risk_state: data.risk_state,
                smoothed_probability: data.smoothed_probability,
                raw_per_expert_scores: data.raw_per_expert_scores,
                audio_quality: data.audio_quality,
                recommended_action: data.recommended_action,
              };
              narrateRemote(fields, (acc) => {
                setReasoning((r) => r.map((l) => (l.id === id ? { ...l, text: acc } : l)));
              })
                .then((full) => {
                  setReasoning((r) =>
                    r.map((l) => (l.id === id ? { ...l, text: full || local.text, streaming: false } : l))
                  );
                })
                .catch(() => {
                  // network / upstream / no-key → fall back to the grounded local line
                  setReasoning((r) =>
                    r.map((l) => (l.id === id ? { ...l, text: local.text, streaming: false } : l))
                  );
                });
            } else {
              setReasoning((r) => [...r, { id, text: local.text, kind: local.kind, streaming: false }]);
            }
          }
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        // Grounded closing line based on the final window we actually saw.
        const v = verdictLine(lastMsgRef.current);
        if (v) {
          setReasoning((r) => [...r, { id: ++lineIdRef.current, text: v.text, kind: v.kind, streaming: false }]);
        }
        stopRecording();
      };

      ws.onerror = (e) => {
        setError('WebSocket error occurred.');
        stopRecording();
      };

    } catch (err) {
      setError(err.message || 'Failed to start microphone or connect to backend.');
      setStatus('disconnected');
      stopRecording();
    }
  };

  const stopRecording = useCallback(() => {
    setIsRecording(false);
    setLatestData(null);
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    // Mux the captured frames into an in-memory WAV for playback. Skip on
    // unmount (nothing to show) and when nothing was captured.
    const chunks = recordedChunksRef.current;
    recordedChunksRef.current = [];
    if (mountedRef.current && chunks.length > 0) {
      const pcm = concatFloat32(chunks);
      if (pcm.length > 0) {
        const blob = encodeWavFromFloat32(pcm, recordSampleRateRef.current);
        setRecordingUrlSafe(URL.createObjectURL(blob));
        setRecordingSpoofed(sessionUsedSpoofRef.current);
        // Freeze the detector's peak call for the playback card (null if no
        // window was ever scored — e.g. the session was all silence).
        setRecordingVerdict(sessionPeakRef.current.prob >= 0 ? { ...sessionPeakRef.current } : null);
      }
    }

    setStatus((prev) => prev !== 'disconnected' ? 'disconnected' : prev);
  }, [setRecordingUrlSafe]);

  // Cleanup on unmount
  useEffect(() => {
    // Set on (re)mount so React 18 StrictMode's setup→cleanup→setup dance leaves
    // mountedRef true; without this the first cleanup pins it false for good and
    // stopRecording would silently skip muxing the WAV.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
      stopRecording();
    };
  }, [stopRecording]);

  // Flip the live spoof on/off. Mirrors relay-service CallSession.set_spoof:
  // reset the DSP state, then re-send `start` so the backend rebuilds its
  // resampler + ring buffer + EMA and the verdict flips fast (~1.2s) instead of
  // dragging the old ~5s smoothing tail across the transition.
  const toggleSpoof = useCallback(() => {
    const next = !spoofEnabledRef.current;
    spoofEnabledRef.current = next;
    setSpoofEnabled(next);
    // Remember that this session's audio was warped, so the playback card can
    // show the provenance ("Voice-changer was on") alongside — but distinct
    // from — the detector's own peak verdict once monitoring stops.
    if (next) sessionUsedSpoofRef.current = true;

    const node = workletNodeRef.current;
    if (node) {
      node.port.postMessage({ type: 'reset' });
      node.port.postMessage({ type: 'config', spoofEnabled: next, profile: spoofProfileRef.current });
    }

    const ws = wsRef.current;
    const ctx = audioContextRef.current;
    if (ws && ws.readyState === WebSocket.OPEN && ctx) {
      ws.send(JSON.stringify({
        type: 'start',
        sample_rate: ctx.sampleRate,
        encoding: 'pcm_f32le',
        channels: 1,
      }));
    }
  }, []);

  const changeSpoofProfile = useCallback((name) => {
    spoofProfileRef.current = name;
    setSpoofProfile(name);
    const node = workletNodeRef.current;
    if (node) {
      node.port.postMessage({ type: 'config', spoofEnabled: spoofEnabledRef.current, profile: name });
    }
  }, []);

  // Union of expert keys seen across the buffered windows. Deriving from every
  // row (not just scores[0]) means the lines still appear when the first windows
  // were "collecting" and carried no per-expert scores yet.
  const expertKeys = Array.from(
    new Set(scores.flatMap((s) => Object.keys(s.expert_probs || {})))
  );

  const StatusIcon = {
    low: CheckCircle,
    uncertain: HelpCircle,
    high: AlertTriangle,
    collecting: Activity,
    unavailable: Activity
  }[currentState] || Activity;

  return (
    <div className="p-4 md:p-8 max-w-6xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-semibold tracking-tight text-zinc-100">Live Voice Monitor</h1>
          <p className="text-zinc-400 mt-1 text-sm md:text-base">Real-time analysis of microphone input for voice cloning artifacts.</p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-zinc-500">Status:</span>
            <Badge variant={status === 'connected' ? 'success' : status === 'connecting' ? 'warning' : 'default'}>
              {status}
            </Badge>
          </div>
          
          {!isRecording ? (
            <Button onClick={connectAndStart} variant="primary">
              <Mic size={16} /> Start Monitoring
            </Button>
          ) : (
            <Button onClick={stopRecording} variant="danger">
              <MicOff size={16} /> Stop Monitoring
            </Button>
          )}
        </div>
      </div>

      {error && (
        <Card className="border-red-500/20 bg-red-500/5">
          <CardContent className="flex items-center gap-3 text-red-400">
            <AlertTriangle className="h-5 w-5" />
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      {isRecording && (
        <Card className={spoofEnabled
          ? 'border-red-500/40 bg-red-500/[0.06] shadow-[0_0_28px_-6px_rgba(239,68,68,0.55)] transition-shadow'
          : 'border-zinc-800 transition-shadow'}>
          <CardContent className="flex flex-col sm:flex-row sm:items-center gap-4 py-4">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${spoofEnabled ? 'bg-red-500 animate-pulse' : 'bg-zinc-600'}`} />
                <p className="text-sm font-semibold text-zinc-200">Attacker Console</p>
                {spoofEnabled && (
                  <span className="text-[10px] uppercase font-bold tracking-wider bg-red-500/10 text-red-400 border border-red-500/30 px-2 py-0.5 rounded">
                    Spoofing Live
                  </span>
                )}
              </div>
              <p className="text-xs text-zinc-500 mt-1">
                {spoofEnabled
                  ? 'Your mic is being warped live with the MockRVC clone chain — pitch + formant shift plus injected neural-vocoder artifacts — the same recipe as the two-phone demo.'
                  : 'Streaming your real voice. Flip the switch to warp it live and see the detector react.'}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <select
                value={spoofProfile}
                onChange={(e) => changeSpoofProfile(e.target.value)}
                className="bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-zinc-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                aria-label="Spoof voice profile"
              >
                {Object.entries(SPOOF_PROFILE_LABELS).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>

              <Button onClick={toggleSpoof} variant={spoofEnabled ? 'danger' : 'secondary'}>
                {spoofEnabled ? <ZapOff size={16} /> : <Zap size={16} />}
                {spoofEnabled ? 'Stop Spoofing' : 'Talk as Teammate 3 (Spoof)'}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className={`md:col-span-1 flex flex-col justify-center items-center p-6 md:p-8 text-center min-h-[250px] md:min-h-[300px] transition-shadow ${spoofEnabled ? 'border-red-500/40 shadow-[0_0_28px_-8px_rgba(239,68,68,0.5)]' : ''}`}>
          {isRecording ? (
             <>
               <div className="relative mb-6">
                 {/* Simple pulse ring */}
                 <div className="absolute inset-0 rounded-full bg-zinc-700 animate-ping opacity-20"></div>
                 <div className="h-20 w-20 bg-zinc-800 rounded-full flex items-center justify-center text-zinc-300 relative z-10 border border-zinc-700">
                   <StatusIcon size={32} className={currentState === 'high' ? 'text-red-500' : currentState === 'low' ? 'text-emerald-500' : 'text-zinc-400'} />
                 </div>
               </div>
               
               <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-widest mb-2">Current State</h2>
               <div className="mb-2 flex justify-center items-center gap-2">
                 <RiskBadge state={currentState || 'collecting'} />
                 {(() => {
                    if (!latestData || !currentState || currentState === 'collecting') return null;
                    const wavlmProb = latestData.scores?.wavlm?.probability;
                    if (typeof wavlmProb !== 'number') return null;
                    if ((currentState === 'high' || currentState === 'uncertain') && wavlmProb < 0.35) {
                      return <span className="text-[10px] uppercase font-bold tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500/20 px-2 py-0.5 rounded flex items-center gap-1"><AlertTriangle size={10} /> Disagreement</span>;
                    }
                    if ((currentState === 'low') && wavlmProb > 0.65) {
                      return <span className="text-[10px] uppercase font-bold tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500/20 px-2 py-0.5 rounded flex items-center gap-1"><AlertTriangle size={10} /> Disagreement</span>;
                    }
                    return null;
                 })()}
               </div>
               {currentState && currentState !== 'collecting' && (
                 <p className="text-4xl font-bold text-zinc-100 mt-4">
                   {(currentProb * 100).toFixed(1)}%
                 </p>
               )}
               
               <div className="mt-8 w-full">
                 <p className="text-xs text-zinc-500 mb-2 uppercase tracking-wider">Live Audio Signal</p>
                 <canvas 
                   ref={canvasRef} 
                   width={300} 
                   height={60} 
                   className="w-full h-16 rounded-md bg-zinc-950/50 border border-zinc-800"
                 />
               </div>
             </>
          ) : (
             <div className="flex flex-col items-center gap-4 opacity-50">
               <MicOff size={48} className="text-zinc-600" />
               <p className="text-zinc-400 font-medium">Microphone is offline</p>
             </div>
          )}
        </Card>

        <Card className="md:col-span-2">
          <div className="p-6 pb-0 flex justify-between items-center">
            <h2 className="text-lg font-semibold tracking-tight">Real-time Probability Stream</h2>
            {latestData && (
              <Badge variant="default" className="bg-zinc-800/50 animate-pulse border-zinc-700">
                Processed Windows: {latestData.sequence_number}
              </Badge>
            )}
          </div>
          <CardContent>
            <div className="h-64 w-full">
              {scores.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={scores} margin={{ top: 5, right: 0, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis 
                      dataKey="time" 
                      stroke="#71717a" 
                      tick={{ fill: '#71717a', fontSize: 12 }} 
                      tickMargin={10}
                    />
                    <YAxis 
                      stroke="#71717a" 
                      tick={{ fill: '#71717a', fontSize: 12 }}
                      domain={[0, 1]}
                      tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                    />
                    <ReferenceLine y={0.35} stroke="#10b981" strokeDasharray="3 3" opacity={0.3} />
                    <ReferenceLine y={0.65} stroke="#ef4444" strokeDasharray="3 3" opacity={0.3} />
                    
                    {/* Render a line for each expert dynamically */}
                    {expertKeys.map(expert => {
                      const color = {
                        wavlm: "#3b82f6",        // blue  · Expert 1 WavLM
                        hybrid: "#ec4899",       // pink  · LFCC-LCNN hybrid family
                        hybrid_br: "#ec4899",
                        hybrid_maxbr: "#ec4899",
                        hybrid_nc: "#f472b6",
                      }[expert] || "#8b5cf6";
                      return (
                        <Line
                          key={expert}
                          type="monotone"
                          dataKey={(d) => d.expert_probs?.[expert] ?? null}
                          name={expert}
                          stroke={color}
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                          connectNulls
                          activeDot={{ r: 4, fill: color }}
                        />
                      );
                    })}
                    <Line 
                      type="monotone" 
                      dataKey="lr_prob" 
                      name="LR Fusion"
                      stroke="#f97316" 
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4, fill: '#f97316' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="prob" 
                      name="Overall Fusion"
                      stroke="#f4f4f5" 
                      strokeWidth={3}
                      dot={false}
                      activeDot={{ r: 6, fill: '#f4f4f5' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full w-full flex items-center justify-center border-2 border-dashed border-zinc-800 rounded-lg text-zinc-500">
                  {isRecording ? 'Waiting for scores...' : 'Start monitoring to see data'}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {!isRecording && recordingUrl && (
        <Card className={
          recordingVerdict?.state === 'high' ? 'border-red-500/30'
          : recordingVerdict?.state === 'uncertain' ? 'border-amber-500/30'
          : recordingVerdict?.state === 'low' ? 'border-emerald-500/20'
          : 'border-zinc-800'
        }>
          <CardHeader
            title="Session Playback"
            description="Exactly the audio the detector analyzed this session. Kept in memory only — it clears the moment you start the next session, and nothing is saved to disk."
          />
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              {/* Provenance — what the OPERATOR did to the audio. This is NOT a
                  detector verdict; the voice-changer being on says nothing about
                  what the model concluded. */}
              {recordingSpoofed ? (
                <span className="text-[10px] uppercase font-bold tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded flex items-center gap-1">
                  <Zap size={10} /> Voice-changer was on
                </span>
              ) : (
                <span className="text-[10px] uppercase font-bold tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded flex items-center gap-1">
                  <Mic size={10} /> Real voice only
                </span>
              )}
              {/* The detector's own peak call over the session (the verdict). */}
              {recordingVerdict ? (
                <span className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                  <span className="uppercase tracking-wider">Detector peak</span>
                  <RiskBadge state={recordingVerdict.state} />
                  <span className="tabular-nums text-zinc-300">{(recordingVerdict.prob * 100).toFixed(1)}%</span>
                </span>
              ) : (
                <span className="text-[11px] text-zinc-500">No windows scored this session</span>
              )}
            </div>
            <p className="text-xs text-zinc-500">
              {recordingSpoofed
                ? (recordingVerdict && recordingVerdict.state !== 'high'
                    ? "You ran this through the voice-changer, but the detector never reached a spoof verdict this session — a browser DSP disguise doesn't carry the neural-synthesis cues these models flag. Play it back to hear exactly what the detector scored."
                    : "This is the warped voice the detector scored — play it back to hear what it heard.")
                : "Play it back to hear exactly what the detector scored."}
            </p>
            <audio
              key={recordingUrl}
              src={recordingUrl}
              controls
              controlsList="nodownload noplaybackrate"
              className="w-full"
            />
          </CardContent>
        </Card>
      )}

      {(isRecording || reasoning.length > 0) && (
        <Card>
          <CardHeader title="Live Reasoning" description="Narration of the detector's own per-window numbers" />
          <CardContent>
            <ReasoningLog reasoning={reasoning} />
          </CardContent>
        </Card>
      )}

      {latestData && (
        <Card>
          <CardHeader title="Detailed Output & Expert Analysis" />
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
              <div>
                <p className="text-sm text-zinc-400 mb-1">Audio Quality</p>
                {latestData.audio_quality === 'silence' ? (
                  <Badge variant="warning">Silence Detected</Badge>
                ) : latestData.audio_quality === 'clipping' ? (
                  <Badge variant="danger">Audio Clipping</Badge>
                ) : latestData.audio_quality ? (
                  <Badge variant="warning">{String(latestData.audio_quality).replace(/_/g, ' ')}</Badge>
                ) : (
                  <Badge variant="success">Optimal</Badge>
                )}
              </div>

              <div>
                <p className="text-sm text-zinc-400 mb-1">Recommended Action</p>
                <Badge variant={latestData.risk_state === 'low' ? 'success' : latestData.risk_state === 'uncertain' ? 'warning' : latestData.risk_state === 'high' ? 'danger' : 'default'}>
                  {latestData.recommended_action || 'Awaiting audio'}
                </Badge>
              </div>
            </div>

            {/* One card per expert, each showing ITS OWN calibrated probability.
                Every expert has a separate Platt fit, so these are comparable as
                verdicts but are not the same number on the same scale. */}
            {latestData.scores && Object.keys(latestData.scores).length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(latestData.scores).map(([model, scoreData]) => {
                  const prob = scoreData?.probability;
                  const hasProb = typeof prob === 'number' && Number.isFinite(prob);
                  const st = scoreData?.risk_state;
                  return (
                    <div key={model} className="rounded-lg border border-zinc-800 p-4">
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <p className="text-sm text-zinc-300">{scoreData?.label || model}</p>
                        <Badge variant={st === 'low' ? 'success' : st === 'uncertain' ? 'warning' : st === 'high' ? 'danger' : 'default'}>
                          {st ? st.toUpperCase() : 'N/A'}
                        </Badge>
                      </div>
                      <p className="text-2xl font-semibold text-zinc-100">
                        {hasProb ? `${(prob * 100).toFixed(1)}%` : 'N/A'}
                        <span className="text-sm font-normal text-zinc-500"> synthetic</span>
                      </p>
                      <p className="text-xs text-zinc-500 mt-2">
                        logit {typeof scoreData?.logit === 'number' ? scoreData.logit.toFixed(2) : 'N/A'}
                        {scoreData?.model_version ? ` · ${scoreData.model_version}` : ''}
                      </p>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-zinc-500">
                No expert scores in this window (audio quality gate or still collecting).
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
