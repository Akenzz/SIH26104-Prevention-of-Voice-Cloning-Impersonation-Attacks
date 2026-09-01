import { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Spinner } from '../components/ui/Spinner';
import { Server, Activity, Cpu, Settings2 } from 'lucide-react';

export default function Settings() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await axios.get('/health');
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
      <div className="flex h-full items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="p-4 md:p-8 max-w-5xl mx-auto space-y-6 pb-20 md:pb-8">
      <div>
        <h1 className="text-xl md:text-2xl font-semibold tracking-tight text-zinc-100">System Health & Config</h1>
        <p className="text-zinc-400 mt-1 text-sm md:text-base">Live status of the Voice Integrity backend services.</p>
      </div>

      {error && (
        <Card className="border-red-500/20 bg-red-500/5">
          <CardContent className="flex items-center gap-3 text-red-400">
            <Activity className="h-5 w-5" />
            <p>Backend Connection Failed: {error}</p>
          </CardContent>
        </Card>
      )}

      {health && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader 
              title="Connection Status" 
              action={<Badge variant="success">Online</Badge>}
            />
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-zinc-800/50">
                <span className="text-zinc-400 flex items-center gap-2"><Server size={16}/> Backend URL</span>
                <span className="text-zinc-100 font-mono text-sm">proxied</span>
              </div>
              <div className="flex justify-between items-start py-2 border-b border-zinc-800/50 gap-4">
                <span className="text-zinc-400 flex items-center gap-2 shrink-0"><Cpu size={16}/> Active Experts</span>
                <div className="flex flex-wrap gap-2 justify-end">
                  {(health.expert_details || (health.experts || []).map((n) => ({ name: n, label: n }))).map((e) => (
                    <Badge key={e.name} variant={e.is_decision_expert ? 'success' : 'default'}>
                      {e.label || e.name}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-zinc-800/50">
                <span className="text-zinc-400 flex items-center gap-2"><Settings2 size={16}/> Decision Expert</span>
                <span className="text-zinc-100 font-mono text-sm">{health.decision_expert || '—'}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-zinc-400 flex items-center gap-2"><Settings2 size={16}/> Fusion Mode</span>
                <span className="text-zinc-100 capitalize">{health.fusion_mode}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader title="Pipeline Configuration" />
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-zinc-800/50">
                <span className="text-zinc-400">Window Size</span>
                <span className="text-zinc-100">{health.window_sec}s</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-zinc-800/50">
                <span className="text-zinc-400">Hop Size</span>
                <span className="text-zinc-100">{health.hop_sec}s</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-zinc-800/50">
                <span className="text-zinc-400">Target Sample Rate</span>
                <span className="text-zinc-100">{health.target_sample_rate} Hz</span>
              </div>
              <div className="flex flex-col gap-1 py-2">
                <span className="text-zinc-400 text-xs uppercase tracking-wider">Versions</span>
                <div className="text-xs font-mono text-zinc-500 mt-2 space-y-1">
                  <div>Policy: {health.threshold_version}</div>
                  <div>Calibrator (decision): {health.calibrator_version}</div>
                  <div>Fusion: {health.fusion_version}</div>
                  {(health.expert_details || []).map((e) => (
                    <div key={e.name}>Calibrator ({e.name}): {e.calibrator_version || '—'}</div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
