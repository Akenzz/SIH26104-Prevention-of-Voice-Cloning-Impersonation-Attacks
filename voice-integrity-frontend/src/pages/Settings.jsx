import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_CONFIG } from '../config';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Spinner } from '../components/ui/Spinner';
import { Server, Activity, Settings2, Cpu, AudioLines, GitCommit } from 'lucide-react';

export default function Settings() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await axios.get(API_CONFIG.BACKEND_URL + '/health');
        setHealth(res.data);
        setError(null);
      } catch (err) {
        setError(err.message || 'Failed to connect to backend.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !health) {
    return (
      <div className="flex h-full items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <Spinner size="lg" />
          <p className="text-zinc-400 font-medium">Connecting to Analysis Engine...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-8 max-w-5xl mx-auto space-y-6 pb-20 md:pb-8">
      {/* Header Section */}
      <div className="border-b border-zinc-800 pb-5 mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-zinc-100 flex items-center gap-3">
            <Settings2 className="text-zinc-400" size={28} />
            System Status & Configuration
          </h1>
          <p className="text-zinc-400 mt-2 text-sm md:text-base max-w-2xl">
            Live diagnostic overview of the Voice Integrity backend. This shows the active AI models, analysis parameters, and connection status for the judges.
          </p>
        </div>
        <div>
          {error ? (
            <Badge variant="destructive" className="px-4 py-2 text-sm flex items-center gap-2 bg-red-900/50 text-red-200 border-red-800">
              <Activity size={16} /> System Offline
            </Badge>
          ) : (
            <Badge variant="success" className="px-4 py-2 text-sm flex items-center gap-2 bg-green-900/40 text-green-300 border-green-800">
              <Server size={16} /> Engine Online
            </Badge>
          )}
        </div>
      </div>

      {error && (
        <Card className="border-red-500/20 bg-red-500/5">
          <CardContent className="flex items-center gap-3 text-red-400 py-4">
            <Activity className="h-6 w-6 shrink-0" />
            <div>
              <p className="font-semibold">Backend Connection Failed</p>
              <p className="text-sm opacity-90">{error}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {health && (
        <div className="space-y-6">
          
          {/* Core Engine Configuration */}
          <Card className="bg-white/5 border-zinc-700 shadow-sm">
            <CardHeader 
              title={<span className="flex items-center gap-2 text-lg"><Cpu className="text-zinc-400" size={20}/> AI Analysis Engine</span>}
              description="Details on the machine learning models currently evaluating the audio."
            />
            <CardContent className="space-y-0 divide-y divide-zinc-800/50">
              
              <div className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-zinc-200 font-medium block">Loaded AI Models</span>
                  <span className="text-zinc-500 text-sm block">The deepfake detection algorithms currently active in memory.</span>
                </div>
                <div className="flex flex-wrap gap-2 md:justify-end">
                  {(health.expert_details || (health.experts || []).map((n) => ({ name: n, label: n }))).map((e) => (
                    <Badge key={e.name} variant="secondary" className="bg-zinc-800 text-zinc-300 border-zinc-700">
                      {e.label || e.name}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-zinc-200 font-medium block">Ensemble Method</span>
                  <span className="text-zinc-500 text-sm block">How the system combines multiple AI predictions into one final score.</span>
                </div>
                <Badge variant="outline" className="text-zinc-300 border-zinc-700 bg-zinc-800/50 uppercase tracking-wider text-xs">
                  {health.fusion_mode?.replace('_', ' ')}
                </Badge>
              </div>

              <div className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-zinc-200 font-medium block">Server Endpoint</span>
                  <span className="text-zinc-500 text-sm block">The URL where the processing backend is hosted.</span>
                </div>
                <span className="text-zinc-400 font-mono text-sm bg-zinc-800/40 px-2 py-1 rounded">
                  {API_CONFIG.BACKEND_URL}
                </span>
              </div>

            </CardContent>
          </Card>

          {/* Audio Pipeline Config */}
          <Card className="bg-white/5 border-zinc-700 shadow-sm">
            <CardHeader 
              title={<span className="flex items-center gap-2 text-lg"><AudioLines className="text-zinc-400" size={20}/> Audio Processing Pipeline</span>} 
              description="Parameters governing how audio is sliced and fed into the AI models."
            />
            <CardContent className="space-y-0 divide-y divide-zinc-800/50">
              
              <div className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-zinc-200 font-medium block">Analysis Segment Size</span>
                  <span className="text-zinc-500 text-sm block">How much audio is analyzed in one pass.</span>
                </div>
                <span className="text-zinc-300 font-medium">{health.window_sec} seconds</span>
              </div>
              
              <div className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-zinc-200 font-medium block">Update Frequency</span>
                  <span className="text-zinc-500 text-sm block">How often the analysis timeline updates.</span>
                </div>
                <span className="text-zinc-300 font-medium">Every {health.hop_sec} seconds</span>
              </div>
              
              <div className="py-4 flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-zinc-200 font-medium block">Audio Quality Standard</span>
                  <span className="text-zinc-500 text-sm block">The frequency format used for processing.</span>
                </div>
                <span className="text-zinc-300 font-medium">{health.target_sample_rate} Hz</span>
              </div>

            </CardContent>
          </Card>

          {/* Version Tracking */}
          <Card className="bg-white/5 border-zinc-700 shadow-sm">
            <CardHeader 
              title={<span className="flex items-center gap-2 text-lg"><GitCommit className="text-zinc-400" size={20}/> System Versions</span>} 
              description="Version control for the active rules and models."
            />
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                
                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                  <span className="text-zinc-400 text-xs font-semibold uppercase tracking-wider block mb-1">
                    Security Rules
                  </span>
                  <span className="text-zinc-500 text-[11px] block mb-2 line-clamp-2">Rules defining the thresholds for Authentic vs. AI-Generated.</span>
                  <span className="text-zinc-300 font-mono text-xs break-all">
                    {health.threshold_version}
                  </span>
                </div>

                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                  <span className="text-zinc-400 text-xs font-semibold uppercase tracking-wider block mb-1">
                    Scoring Engine
                  </span>
                  <span className="text-zinc-500 text-[11px] block mb-2 line-clamp-2">Translates raw model output into a 0-100% confidence score.</span>
                  <span className="text-zinc-300 font-mono text-xs break-all">
                    {health.calibrator_version}
                  </span>
                </div>

                <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                  <span className="text-zinc-400 text-xs font-semibold uppercase tracking-wider block mb-1">
                    Consensus Strategy
                  </span>
                  <span className="text-zinc-500 text-[11px] block mb-2 line-clamp-2">How multiple AI models agree on a final decision.</span>
                  <span className="text-zinc-300 font-mono text-xs break-all">
                    {health.fusion_version}
                  </span>
                </div>

                {/* Expert Specific Calibrators */}
                {(health.expert_details || []).map((e) => (
                  <div key={e.name} className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-4">
                    <span className="text-zinc-400 text-xs font-semibold uppercase tracking-wider block mb-1">
                      {e.label || e.name} Engine
                    </span>
                    <span className="text-zinc-500 text-[11px] block mb-2 line-clamp-2">The active version of this specific AI model.</span>
                    <span className="text-zinc-300 font-mono text-xs break-all">
                      {e.calibrator_version || 'N/A'}
                    </span>
                  </div>
                ))}
                
              </div>
            </CardContent>
          </Card>
          
        </div>
      )}
    </div>
  );
}
