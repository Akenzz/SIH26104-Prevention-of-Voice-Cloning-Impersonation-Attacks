import { Link } from 'react-router-dom';
import { Radio, FileAudio, Settings, ShieldCheck, Activity } from 'lucide-react';
import { Card, CardHeader, CardContent } from '../components/ui/Card';

export default function Dashboard() {
  return (
    <div className="p-4 md:p-8 max-w-6xl mx-auto space-y-6 md:space-y-8">
      <div className="flex items-center gap-4 border-b border-zinc-800 pb-4 md:pb-6">
        <div className="w-10 h-10 md:w-12 md:h-12 rounded-xl bg-zinc-100 flex items-center justify-center text-zinc-950 shrink-0">
          <ShieldCheck size={24} className="md:w-7 md:h-7" strokeWidth={2.5} />
        </div>
        <div>
          <h1 className="text-xl md:text-3xl font-bold tracking-tight text-zinc-100">Voice Integrity Platform</h1>
          <p className="text-zinc-400 mt-1 text-sm md:text-base">Enterprise-grade deepfake and voice cloning detection.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link to="/live-monitor" className="block group">
          <Card className="h-full transition-colors group-hover:border-zinc-700 group-hover:bg-zinc-800/20">
            <CardContent className="p-6 md:p-8 flex flex-col items-center text-center gap-4">
              <div className="h-12 w-12 md:h-16 md:w-16 rounded-full bg-zinc-900 flex items-center justify-center text-zinc-300 group-hover:text-zinc-100 transition-colors">
                <Radio size={24} className="md:w-8 md:h-8" />
              </div>
              <div>
                <h2 className="text-lg md:text-xl font-semibold text-zinc-100 mb-2">Live Monitor</h2>
                <p className="text-zinc-400 text-sm md:text-base">
                  Connect to the real-time WebSocket stream to analyze microphone audio continuously. 
                  Ideal for live call center integrations and real-time alerts.
                </p>
              </div>
            </CardContent>
          </Card>
        </Link>

        <Link to="/file-analysis" className="block group">
          <Card className="h-full transition-colors group-hover:border-zinc-700 group-hover:bg-zinc-800/20">
            <CardContent className="p-6 md:p-8 flex flex-col items-center text-center gap-4">
              <div className="h-12 w-12 md:h-16 md:w-16 rounded-full bg-zinc-900 flex items-center justify-center text-zinc-300 group-hover:text-zinc-100 transition-colors">
                <FileAudio size={24} className="md:w-8 md:h-8" />
              </div>
              <div>
                <h2 className="text-lg md:text-xl font-semibold text-zinc-100 mb-2">Batch Processing</h2>
                <p className="text-zinc-400 text-sm md:text-base">
                  Upload complete audio files (WAV, FLAC, MP3) for full timeline analysis.
                  Visualizes risk probability across the entire duration of the recording.
                </p>
              </div>
            </CardContent>
          </Card>
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-3">
          <CardHeader title="System Architecture" />
          <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-zinc-400">
            <div className="flex items-start gap-3">
              <div className="mt-1 text-emerald-500"><Activity size={18}/></div>
              <div>
                <h4 className="text-zinc-200 font-medium mb-1">Dual-Expert AI</h4>
                <p>Powered by WavLM Base+ and LFCC-LCNN models for robust spoof detection.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1 text-emerald-500"><Activity size={18}/></div>
              <div>
                <h4 className="text-zinc-200 font-medium mb-1">Platt Calibration</h4>
                <p>Raw logits are calibrated into true probability scores using a Platt scaler fit on ASVspoof19 data.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1 text-emerald-500"><Activity size={18}/></div>
              <div>
                <h4 className="text-zinc-200 font-medium mb-1">Low Latency</h4>
                <p>Optimized 4.0s windowing with 0.5s hop size ensures rapid real-time updates.</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
