import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, FileAudio, AlertCircle, BarChart3, ChevronDown, Sliders } from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge, RiskBadge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Spinner } from '../components/ui/Spinner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const FUSION_MODES = [
  { key: 'lr_fusion',      label: 'LR Fusion (3-Expert)',    dataKey: 'lr_probability',           color: '#a855f7' },
  { key: 'heuristic_avg', label: 'LFCC + SSL Avg',          dataKey: 'heuristic_avg_probability', color: '#f97316' },
  { key: 'wavlm',         label: 'WavLM Only',              dataKey: 'per_expert_probability.wavlm', color: '#3b82f6' },
  { key: 'hybrid',        label: 'LFCC-LCNN Only',          dataKey: 'per_expert_probability.hybrid', color: '#ec4899' },
  { key: 'ssl',           label: 'TakHemlata SSL Only',     dataKey: 'per_expert_probability.ssl',   color: '#eab308' },
];

export default function FileAnalysis() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedExpert, setExpandedExpert] = useState(null);
  const [selectedMode, setSelectedMode] = useState('lr_fusion'); // which signal drives overall line
  const [health, setHealth] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    axios.get('/health').then(r => {
      setHealth(r.data);
      // auto-select whatever the backend is running
      const m = r.data?.fusion_mode;
      if (m && FUSION_MODES.find(f => f.key === m)) setSelectedMode(m);
    }).catch(() => {});
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
      version: result?.model_version?.[expert] || '—',
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
      const res = await axios.post('/predict-file', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to process file.');
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
            className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center transition-colors cursor-pointer ${
              isDragging ? 'border-zinc-500 bg-zinc-800/50' : 'border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/20'
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
            <BarChart3 size={20} className="text-zinc-400"/> Analysis Results
          </h2>

          {/* Mode Selector */}
          <Card>
            <CardContent className="pt-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="flex items-center gap-1.5 text-sm text-zinc-400 mr-2">
                  <Sliders size={14} /> Primary Decision:
                </span>
                {FUSION_MODES.map(m => (
                  <button
                    key={m.key}
                    onClick={() => setSelectedMode(m.key)}
                    className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-all border ${
                      selectedMode === m.key
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
              </div>
              <p className="text-xs text-zinc-600 mt-2">
                Select which model's probability drives the white Overall line on the chart. This is purely visual — the backend decision uses the mode tagged "(backend)".
              </p>
            </CardContent>
          </Card>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="md:col-span-2">
              <CardHeader title="Overall Assessment" />
              <CardContent>
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 sm:gap-6">
                  <div className="flex-1 w-full flex justify-between sm:block">
                    <p className="text-sm text-zinc-400 mb-1">Final Risk State</p>
                    <div className="flex items-center gap-2">
                      <RiskBadge state={result.summary.overall_risk_state} />
                      {(() => {
                        const finalState = result.summary.overall_risk_state;
                        const wavlmProb = result.summary.expert_probabilities?.wavlm;
                        if (typeof wavlmProb !== 'number') return null;
                        if ((finalState === 'spoof' || finalState === 'suspicious') && wavlmProb < 0.35) {
                          return <span className="text-[10px] uppercase font-bold tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500/20 px-2 py-0.5 rounded flex items-center gap-1"><AlertCircle size={10} /> Disagreement</span>;
                        }
                        if (finalState === 'bonafide' && wavlmProb > 0.65) {
                          return <span className="text-[10px] uppercase font-bold tracking-wider bg-amber-500/10 text-amber-500 border border-amber-500/20 px-2 py-0.5 rounded flex items-center gap-1"><AlertCircle size={10} /> Disagreement</span>;
                        }
                        return null;
                      })()}
                    </div>
                    <p className="text-xs text-zinc-600 mt-1">
                      Backend: {health?.decision_label || health?.fusion_mode || '…'}
                    </p>
                  </div>
                  <div className="flex-1 w-full flex justify-between sm:block">
                    <p className="text-sm text-zinc-400 mb-1">Max Probability</p>
                    <p className="text-lg sm:text-2xl font-semibold text-zinc-100">
                      {result.summary.max_probability ? (result.summary.max_probability * 100).toFixed(1) : 0}%
                    </p>
                  </div>
                  <div className="flex-1 w-full flex justify-between sm:block">
                    <p className="text-sm text-zinc-400 mb-1">Peak Location</p>
                    <p className="text-base sm:text-lg font-medium text-zinc-300">
                      Window {result.summary.max_probability_window_index}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader title="Expert Models" description="Each model's own calibrated verdict — click for per-window scores" />
              <CardContent className="space-y-2">
                {expertList().map((e) => {
                  const expert = e.name;
                  const state = e.risk_state;
                  const isOpen = expandedExpert === expert;
                  const stats = isOpen ? expertStats(expert) : null;
                  const hasProb = typeof e.probability === 'number' && Number.isFinite(e.probability);
                  // Only show DECISION badge when the backend is running in single-expert mode for this expert
                  const isActualDecision = health?.fusion_mode === 'single' && e.is_decision_expert;
                  return (
                    <div key={expert} className="border-b border-zinc-800/50 last:border-0">
                      <button
                        type="button"
                        onClick={() => setExpandedExpert(isOpen ? null : expert)}
                        aria-expanded={isOpen}
                        className="w-full flex justify-between items-center py-3 text-left hover:opacity-80 transition-opacity"
                      >
                        <span className="flex items-center gap-2">
                          <ChevronDown
                            size={16}
                            className={`text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                          />
                          <span className="text-sm font-medium text-zinc-300">{e.label || expert}</span>
                          {isActualDecision && (
                            <span className="text-[10px] uppercase tracking-wide text-zinc-500 border border-zinc-700 rounded px-1.5 py-0.5">
                              decision
                            </span>
                          )}
                        </span>
                        <span className="flex items-center gap-3">
                          <span className="text-sm text-zinc-400 tabular-nums">
                            {hasProb ? `${(e.probability * 100).toFixed(1)}%` : '—'}
                          </span>
                          <RiskBadge state={state} />
                        </span>
                      </button>

                      {isOpen && (
                        <div className="pb-3 pl-6 pr-1">
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
                            <p className="text-xs text-zinc-500 py-2">No per-window scores available for this model.</p>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>

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

                    {/* All signals as dim dashed reference lines */}
                    <Line type="monotone" dataKey="per_expert_probability.wavlm"  name="WavLM"        stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="5 5" dot={false} opacity={selectedMode === 'wavlm'  ? 0 : 0.45} />
                    <Line type="monotone" dataKey="per_expert_probability.hybrid" name="LFCC-LCNN"    stroke="#ec4899" strokeWidth={1.5} strokeDasharray="5 5" dot={false} opacity={selectedMode === 'hybrid' ? 0 : 0.45} />
                    <Line type="monotone" dataKey="per_expert_probability.ssl"    name="SSL"          stroke="#eab308" strokeWidth={1.5} strokeDasharray="5 5" dot={false} opacity={selectedMode === 'ssl'    ? 0 : 0.45} />
                    <Line type="monotone" dataKey="lr_probability"                name="LR Fusion"    stroke="#a855f7" strokeWidth={1.5} strokeDasharray="4 4" dot={false} opacity={selectedMode === 'lr_fusion'      ? 0 : 0.45} />
                    <Line type="monotone" dataKey="heuristic_avg_probability"     name="LFCC+SSL Avg" stroke="#f97316" strokeWidth={1.5} strokeDasharray="4 4" dot={false} opacity={selectedMode === 'heuristic_avg'  ? 0 : 0.45} />

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
