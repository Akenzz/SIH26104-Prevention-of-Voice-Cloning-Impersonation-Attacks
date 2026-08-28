import { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, FileAudio, AlertCircle, BarChart3, Activity } from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge, RiskBadge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Spinner } from '../components/ui/Spinner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function FileAnalysis() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

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
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="md:col-span-2">
              <CardHeader title="Overall Assessment" />
              <CardContent>
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 sm:gap-6">
                  <div className="flex-1 w-full flex justify-between sm:block">
                    <p className="text-sm text-zinc-400 mb-1">Final Risk State</p>
                    <RiskBadge state={result.summary.overall_risk_state} />
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
              <CardHeader title="Expert Models" />
              <CardContent className="space-y-3">
                {result.summary.expert_risk_states && Object.entries(result.summary.expert_risk_states).map(([expert, state]) => (
                  <div key={expert} className="flex justify-between items-center pb-3 border-b border-zinc-800/50 last:border-0 last:pb-0">
                    <span className="text-sm font-medium text-zinc-300 uppercase">{expert}</span>
                    <RiskBadge state={state} />
                  </div>
                ))}
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
                      formatter={(val, name) => {
                        if (name === 'calibrated_probability') return [`${(val * 100).toFixed(2)}%`, 'Probability'];
                        return [val, name];
                      }}
                    />
                    <ReferenceLine y={0.35} stroke="#10b981" strokeDasharray="3 3" opacity={0.3} />
                    <ReferenceLine y={0.65} stroke="#ef4444" strokeDasharray="3 3" opacity={0.3} />
                    <Line 
                      type="monotone" 
                      dataKey="calibrated_probability" 
                      stroke="#f4f4f5" 
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 6, fill: '#f4f4f5' }}
                    />
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
