import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, FileAudio, AlertCircle, BarChart3, ChevronDown, Sliders, CheckCircle, XCircle, AlertTriangle, Activity } from 'lucide-react';
import { API_CONFIG } from '../config';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge, RiskBadge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Spinner } from '../components/ui/Spinner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const FUSION_MODES = [
  { key: 'heuristic_avg', label: 'Fused (Expert-1 + Expert-2)', dataKey: 'weighted_probability', color: '#8b5cf6' },
  { key: 'wavlm', label: 'WavLM (best_model_v6.pt)', dataKey: 'per_expert_probability.wavlm', color: '#3b82f6' },
  { key: 'hybrid_maxbr', label: 'LFCC-LCNN Only', dataKey: 'per_expert_probability.hybrid_maxbr', color: '#ec4899' },
];

export default function FileAnalysis() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedExpert, setExpandedExpert] = useState(null);
  const [selectedMode, setSelectedMode] = useState('heuristic_avg'); // which signal drives overall line
  const [health, setHealth] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    axios.get(API_CONFIG.BACKEND_URL + '/health').then(r => {
      setHealth(r.data);
      // auto-select whatever the backend is running
      const m = r.data?.fusion_mode;
      if (m && FUSION_MODES.find(f => f.key === m)) setSelectedMode(m);
    }).catch(() => { });
  }, []);

  // Aggregate a single expert's raw per-window logits into displayable stats.
  // The backend reports raw logits (higher = more spoof-like) per window in
  // `windows[].raw_per_expert_scores`; the model_version map carries its label.
  const expertStats = (expert) => {
    const logits = (result?.windows || [])
      .map((w) => w.raw_per_expert_scores?.[expert])
      .filter((v) => typeof v === 'number');
    if (logits.length === 0) return null;
    const max = Math.max(...logits);
    const min = Math.min(...logits);
    const mean = logits.reduce((a, b) => a + b, 0) / logits.length;
    return {
      count: logits.length,
      max,
      min,
      mean,
      version: result?.model_version?.[expert] || 'N/A',
    };
  };

  // `summary.experts` is the backend's own list (label, own calibrated
  // probability, own band, decision-expert flag). Falling back to the older
  // expert_risk_states map keeps this working against an older backend.
  const expertList = () => {
    if (Array.isArray(result?.summary?.experts) && result.summary.experts.length > 0) {
      return result.summary.experts;
    }
    return Object.entries(result?.summary?.expert_risk_states || {}).map(([name, state]) => ({
      name,
      label: name,
      risk_state: state,
      probability: null,
      is_decision_expert: false,
    }));
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const processFile = async () => {
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(API_CONFIG.BACKEND_URL + '/predict-file', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let tempResult = { windows: [], summary: null };
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');

        // Keep the last partial event in the buffer
        buffer = events.pop() || '';

        for (const event of events) {
          const lines = event.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (!dataStr) continue;
              try {
                const data = JSON.parse(dataStr);
                if (data.event === 'window_scored' || data.event === 'skipped' || data.event === 'quality_fail') {
                  tempResult.windows = [...tempResult.windows, data];

                  if (data.event === 'window_scored') {
                    tempResult.summary = {
                      overall_risk_state: 'collecting',
                      weighted_spoof_probability: data.weighted_probability,
                      confidence_level: '...',
                      agreement: '...',
                      spoof_windows_count: tempResult.windows.filter(w => w.weighted_probability > 0.5).length,
                      total_windows_count: tempResult.windows.length,
                      experts: Object.keys(data.per_expert_probability || {}).map(name => ({
                        name,
                        label: name,
                        probability: data.per_expert_probability[name],
                        risk_state: 'collecting'
                      }))
                    };
                  }

                  // Trigger re-render with partial data
                  setResult({ ...tempResult });
                } else if (data.event === 'summary') {
                  tempResult.summary = data;
                  setResult({ ...tempResult });
                }
              } catch (err) {
                console.error('Error parsing SSE JSON:', err, 'Data string:', dataStr);
              }
            }
          }
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to process file.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="p-4 md:p-8 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl md:text-2xl font-semibold tracking-tight text-zinc-100">Batch Processing</h1>
        <p className="text-zinc-400 mt-1 text-sm md:text-base">Upload an audio file to analyze its entire duration for voice cloning.</p>
      </div>

      {/* Upload Zone */}
      <Card>
        <CardContent>
          <div
            className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center transition-colors cursor-pointer ${isDragging ? 'border-zinc-500 bg-zinc-800/50' : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/20'
              }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="audio/*"
              className="hidden"
            />

            {file ? (
              <div className="flex flex-col items-center gap-3">
                <div className="h-16 w-16 rounded-full bg-zinc-800 flex items-center justify-center text-zinc-300">
                  <FileAudio size={32} />
                </div>
                <div className="text-center">
                  <p className="text-zinc-200 font-medium">{file.name}</p>
                  <p className="text-zinc-500 text-sm">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="h-16 w-16 rounded-full bg-zinc-900 flex items-center justify-center text-zinc-500">
                  <UploadCloud size={32} />
                </div>
                <div className="text-center">
                  <p className="text-zinc-300 font-medium">Click or drag audio file here</p>
                  <p className="text-zinc-500 text-sm">Supports WAV, FLAC, MP3</p>
                </div>
              </div>
            )}
          </div>

          <div className="mt-4 flex justify-end gap-3">
            {file && (
              <Button variant="ghost" onClick={() => { setFile(null); setResult(null); }}>
                Clear
              </Button>
            )}
            <Button
              variant="primary"
              onClick={(e) => { e.stopPropagation(); processFile(); }}
              disabled={!file}
              isLoading={isProcessing}
            >
              Analyze Audio
            </Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-red-500/20 bg-red-500/5">
          <CardContent className="flex items-center gap-3 text-red-400">
            <AlertCircle className="h-5 w-5" />
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Results Section */}
      {result && result.summary && (
        <div className="space-y-6">
          <h2 className="text-lg font-medium text-zinc-100 flex items-center gap-2">
            <BarChart3 size={20} className="text-zinc-400" /> Analysis Results
          </h2>

          {/* ── Verdict Hero Card ── */}
          {(() => {
            const s = result.summary;
            const prob = s.weighted_spoof_probability ?? s.max_probability ?? s.final_smoothed_probability ?? 0;
            const pct = (prob * 100).toFixed(1);
            const state = s.overall_risk_state;
            const isFake = state === 'spoof' || state === 'high';
            const isReal = state === 'bonafide' || state === 'low';
            const isCollecting = state === 'collecting';

            const VerdictIcon = isCollecting ? Activity : isFake ? XCircle : isReal ? CheckCircle : AlertTriangle;
            const verdictLabel = isCollecting ? 'ANALYZING...' : isFake ? 'AI GENERATED (SPOOF)' : isReal ? 'HUMAN VOICE (REAL)' : 'UNCLEAR / SUSPICIOUS';
            const verdictColor = isCollecting ? '#3b82f6' : isFake ? '#dc2626' : isReal ? '#16a34a' : '#d97706';
            const barColor = prob > 0.65 ? '#dc2626' : prob > 0.35 ? '#d97706' : '#16a34a';

            const agreementMap = {
              unanimous_spoof: { label: 'Unanimous - All models say AI', color: '#dc2626' },
              unanimous_bonafide: { label: 'Unanimous - All models say HUMAN', color: '#16a34a' },
              majority_spoof: { label: 'Majority vote - Likely AI', color: '#d97706' },
              majority_bonafide: { label: 'Majority vote - Likely HUMAN', color: '#65a30d' },
            };
            const agInfo = agreementMap[s.agreement] || { label: s.agreement || 'N/A', color: '#71717a' };
            const confColor = { high: '#16a34a', medium: '#d97706', low: '#dc2626' }[s.confidence_level] || '#71717a';

            return (
              <Card className="border-zinc-800 bg-zinc-900/40">
                <CardContent className="pt-6 pb-5">
                  {/* Top row: icon + verdict + confidence pill */}
                  <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
                    <div className="flex items-center gap-4">
                      <VerdictIcon size={40} color={verdictColor} strokeWidth={1.5} />
                      <div>
                        <p className="text-2xl font-bold tracking-tight text-zinc-100">{verdictLabel}</p>
                        <p className="text-xs text-zinc-400 mt-0.5">Based on weighted ensemble of 2 expert models</p>
                      </div>
                    </div>
                    <span
                      className="text-xs font-semibold uppercase tracking-widest px-3 py-1.5 rounded border"
                      style={{ color: confColor, borderColor: `${confColor}30`, background: `${confColor}10` }}
                    >
                      {s.confidence_level} confidence
                    </span>
                  </div>

                  {/* Spoof Probability bar */}
                  <div className="mb-5">
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="text-zinc-400">Spoof Probability</span>
                      <span className="font-semibold tabular-nums" style={{ color: barColor }}>{pct}%</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-zinc-800 overflow-hidden">
                      <div
                        className="h-3 rounded-full transition-all duration-700"
                        style={{ width: `${pct}%`, backgroundColor: barColor }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-zinc-600 mt-1">
                      <span>REAL (0%)</span>
                      <span>Threshold 35%</span>
                      <span>FAKE (100%)</span>
                    </div>
                  </div>

                  {/* Stats row */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: 'Expert Agreement', value: agInfo.label, color: agInfo.color, small: true },
                      { label: 'Suspicious Windows', value: `${s.spoof_windows_count ?? 0} / ${s.total_windows_count ?? 0}`, color: '#d4d4d8' },
                      { label: 'Peak Detected At', value: s.peak_time_sec != null ? `${s.peak_time_sec}s` : 'N/A', color: '#d4d4d8' },
                      { label: 'Risk State', value: verdictLabel, color: verdictColor },
                    ].map(({ label, value, color, small }) => (
                      <div key={label} className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-3">
                        <p className="text-xs text-zinc-500 mb-1">{label}</p>
                        <p className={`font-semibold ${small ? 'text-xs leading-tight' : 'text-sm'}`} style={{ color }}>{value}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            );
          })()}

          {/* ── Expert Breakdown ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="md:col-span-2">
              <CardHeader title="Expert Model Breakdown" />
              <CardContent className="space-y-1">
                {expertList().map((e) => {
                  const expert = e.name;
                  const isOpen = expandedExpert === expert;
                  const stats = isOpen ? expertStats(expert) : null;
                  const prob = typeof e.probability === 'number' ? e.probability : null;
                  const pct = prob != null ? (prob * 100).toFixed(1) : null;
                  const barClr = prob > 0.65 ? '#ef4444' : prob > 0.35 ? '#f97316' : '#22c55e';
                  const expertWeights = { wavlm: 25, hybrid_maxbr: 75 };
                  const weight = expertWeights[expert];

                  return (
                    <div key={expert} className="border-b border-zinc-800/50 last:border-0 py-3">
                      <button
                        type="button"
                        onClick={() => setExpandedExpert(isOpen ? null : expert)}
                        className="w-full text-left"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <ChevronDown size={14} className={`text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                            <span className="text-sm font-medium text-zinc-200">{e.label || expert}</span>
                            {weight != null && (
                              <span className="text-[10px] text-zinc-500 border border-zinc-700 rounded px-1.5 py-0.5">{weight}% weight</span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold tabular-nums" style={{ color: prob != null ? barClr : '#71717a' }}>
                              {pct != null ? `${pct}%` : 'N/A'}
                            </span>
                            <RiskBadge state={e.risk_state} />
                          </div>
                        </div>
                        {/* Per-expert probability bar */}
                        {pct != null && (
                          <div className="w-full h-1.5 rounded-full bg-zinc-800 overflow-hidden ml-5">
                            <div className="h-1.5 rounded-full" style={{ width: `${pct}%`, backgroundColor: barClr }} />
                          </div>
                        )}
                      </button>

                      {isOpen && (
                        <div className="mt-3 pl-5">
                          {stats ? (
                            <div className="rounded-md bg-zinc-900/60 border border-zinc-800 p-3 space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-zinc-500">Model version</span>
                                <span className="text-zinc-300 font-mono text-xs">{stats.version}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-zinc-500">Peak logit (most spoof-like)</span>
                                <span className="text-zinc-100 font-medium">{stats.max.toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-zinc-500">Mean logit</span>
                                <span className="text-zinc-300">{stats.mean.toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-zinc-500">Min logit (most bonafide-like)</span>
                                <span className="text-zinc-300">{stats.min.toFixed(3)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-zinc-500">Windows scored</span>
                                <span className="text-zinc-300">{stats.count}</span>
                              </div>
                              <p className="text-[11px] text-zinc-600 pt-1 border-t border-zinc-800">
                                Raw logits: higher = more spoof-like, lower = more bonafide-like.
                              </p>
                            </div>
                          ) : (
                            <p className="text-xs text-zinc-500 py-2">No per-window scores available.</p>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {/* Mode Selector card */}
            <Card>
              <CardHeader title="Graph Display Mode" />
              <CardContent className="space-y-2">
                <p className="text-xs text-zinc-500 mb-3">Select which signal drives the primary white line on the timeline chart.</p>
                {FUSION_MODES.map(m => (
                  <button
                    key={m.key}
                    onClick={() => setSelectedMode(m.key)}
                    className={`w-full px-3 py-2 rounded-lg text-left text-xs font-semibold transition-all border ${selectedMode === m.key
                        ? 'border-transparent text-zinc-950'
                        : 'border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200'
                      }`}
                    style={selectedMode === m.key ? { backgroundColor: m.color } : {}}
                  >
                    {m.label}
                    {health?.fusion_mode === m.key && (
                      <span className="ml-1.5 opacity-70">(backend)</span>
                    )}
                  </button>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* ── Timeline Chart ── */}
          <Card>
            <CardHeader title="Timeline View" description="Risk probability across the entire audio file" />
            <CardContent>
              <div className="h-80 w-full mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={result.windows} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis
                      dataKey="start_time_sec"
                      stroke="#71717a"
                      tick={{ fill: '#71717a', fontSize: 12 }}
                      tickFormatter={(val) => `${val.toFixed(1)}s`}
                    />
                    <YAxis
                      stroke="#71717a"
                      tick={{ fill: '#71717a', fontSize: 12 }}
                      domain={[0, 1]}
                      tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#f4f4f5' }}
                      itemStyle={{ color: '#f4f4f5' }}
                      labelFormatter={(val) => `Time: ${Number(val).toFixed(1)}s`}
                      formatter={(val, name) => [`${(Number(val) * 100).toFixed(1)}%`, name]}
                    />
                    <Legend wrapperStyle={{ paddingTop: '12px', fontSize: '12px', color: '#a1a1aa' }} />
                    <ReferenceLine y={0.35} stroke="#10b981" strokeDasharray="3 3" opacity={0.3} />
                    <ReferenceLine y={0.65} stroke="#ef4444" strokeDasharray="3 3" opacity={0.3} />

                    <Line type="monotone" dataKey="per_expert_probability.wavlm" name="WavLM (MaxBR v6)" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="5 5" dot={false} opacity={selectedMode === 'wavlm' ? 0 : 0.45} />
                    <Line type="monotone" dataKey="per_expert_probability.hybrid_maxbr" name="LFCC-Max" stroke="#ec4899" strokeWidth={1.5} strokeDasharray="5 5" dot={false} opacity={selectedMode === 'hybrid_maxbr' ? 0 : 0.45} />
                    <Line type="monotone" dataKey="weighted_probability" name="Weighted Avg" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4 4" dot={false} opacity={selectedMode === 'heuristic_avg' ? 0 : 0.45} />

                    {/* Selected mode promoted to bold white primary line */}
                    {(() => {
                      const mode = FUSION_MODES.find(m => m.key === selectedMode);
                      if (!mode) return null;
                      return (
                        <Line
                          type="monotone"
                          dataKey={mode.dataKey}
                          name={`▶ ${mode.label} (Primary)`}
                          stroke="#f4f4f5"
                          strokeWidth={3}
                          dot={false}
                          activeDot={{ r: 6, fill: '#f4f4f5' }}
                        />
                      );
                    })()}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

        </div>
      )}
    </div>
  );
}
