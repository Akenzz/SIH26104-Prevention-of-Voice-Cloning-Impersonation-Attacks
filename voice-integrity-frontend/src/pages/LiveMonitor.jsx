import { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, MicOff, Activity, AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge, RiskBadge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { narrate, verdictLine } from '../lib/narrator';
import { narrateRemote } from '../lib/narrateRemote';
import ReasoningLog from '../components/panels/ReasoningLog';

// Inline AudioWorklet processor to capture PCM f32le frames
const workletCode = `
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      // Post the Float32Array to the main thread
      this.port.postMessage(channelData);
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);
`;

export default function LiveMonitor() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  
  const [scores, setScores] = useState([]);
  const [currentState, setCurrentState] = useState(null);
  const [currentProb, setCurrentProb] = useState(0);
  const [latestData, setLatestData] = useState(null);
  const [reasoning, setReasoning] = useState([]);

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

  const connectAndStart = async () => {
    try {
      setError(null);
      setStatus('connecting');

      // 1. Get Microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
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

        const workletNode = new AudioWorkletNode(audioCtx, 'capture-processor');
        workletNodeRef.current = workletNode;

        workletNode.port.onmessage = (e) => {
          const f32Array = e.data;
          // Send raw binary float32 array
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(f32Array.buffer);
          }
        };

        source.connect(workletNode);
        workletNode.connect(audioCtx.destination); // Required for some browsers to keep worklet alive, but we could mute it.
        // Actually, connecting to destination causes feedback. Better to connect to a GainNode with 0 gain.
        const gain = audioCtx.createGain();
        gain.gain.value = 0;
        workletNode.connect(gain);
        gain.connect(audioCtx.destination);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'score') {
          const newScore = {
            time: (data.sequence_number * 0.5).toFixed(1), // Assuming 0.5s hop for display
            prob: data.smoothed_probability,
            lr_prob: data.lr_probability,
            state: data.risk_state,
            expert_probs: data.expert_probabilities || {},
          };
          
          setScores(prev => {
            const next = [...prev, newScore];
            if (next.length > 50) next.shift(); // Keep last 50 points
            return next;
          });
          
          setCurrentState(data.risk_state);
          setCurrentProb(data.smoothed_probability);
          setLatestData(data);

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
    
    setStatus((prev) => prev !== 'disconnected' ? 'disconnected' : prev);
  }, []); // Empty dependency array

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

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

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-1 flex flex-col justify-center items-center p-6 md:p-8 text-center min-h-[250px] md:min-h-[300px]">
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
                    const wavlmProb = latestData.expert_probabilities?.wavlm;
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
                    {scores.length > 0 && scores[0].expert_probs && Object.keys(scores[0].expert_probs).map(expert => {
                      const color = {
                        wavlm: "#3b82f6", // blue
                        hybrid: "#ec4899", // pink
                        ssl: "#eab308" // yellow
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
                        {hasProb ? `${(prob * 100).toFixed(1)}%` : '—'}
                        <span className="text-sm font-normal text-zinc-500"> synthetic</span>
                      </p>
                      <p className="text-xs text-zinc-500 mt-2">
                        logit {typeof scoreData?.logit === 'number' ? scoreData.logit.toFixed(2) : '—'}
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
