import "./styles/app.css";
import { useMonitor } from "@/lib/useMonitor.js";
import { Panel } from "@/components/ui/primitives.jsx";
import StatusRail from "@/components/layout/StatusRail.jsx";
import Disclaimer from "@/components/layout/Disclaimer.jsx";
import SourcePanel from "@/components/panels/SourcePanel.jsx";
import SignalStrip from "@/components/viz/SignalStrip.jsx";
import VerdictPanel from "@/components/panels/VerdictPanel.jsx";
import TimelineChart from "@/components/viz/TimelineChart.jsx";
import ReasoningLog from "@/components/panels/ReasoningLog.jsx";
import ExpertPanel from "@/components/panels/ExpertPanel.jsx";

export default function App() {
  const m = useMonitor();

  return (
    <div className="app">
      <StatusRail current={m.current} source={m.source} running={m.status === "running"} />
      <Disclaimer />

      <main className="console">
        <div className="col col-left">
          <Panel eyebrow="Control deck" title="Signal source">
            <SourcePanel
              source={m.source} setSource={m.setSource}
              scenario={m.scenario} setScenario={m.setScenario}
              inputKind={m.inputKind} setInputKind={m.setInputKind}
              backendBase={m.backendBase} setBackendBase={m.setBackendBase}
              status={m.status} running={m.running} error={m.error} elapsedSec={m.elapsedSec}
              health={m.health} healthError={m.healthError} refreshHealth={m.refreshHealth}
              onStart={m.start} onStop={m.stop}
            />
          </Panel>

          <Panel eyebrow="Input" title="Live waveform">
            <SignalStrip readWaveform={m.readWaveform} running={m.status === "running"} state={m.current?.risk_state} />
          </Panel>
        </div>

        <div className="col col-center">
          <VerdictPanel current={m.current} status={m.status} />

          <Panel eyebrow="Smoothed spoof probability" title="Risk timeline">
            <div className="timeline-box">
              <TimelineChart series={m.series} />
            </div>
          </Panel>
        </div>

        <div className="col col-right">
          <Panel
            eyebrow="Task F · grounded narration"
            title="Live reasoning"
          >
            <ReasoningLog reasoning={m.reasoning} />
          </Panel>

          <Panel eyebrow="Detector" title="Experts & integrity">
            <ExpertPanel current={m.current} stats={m.stats} />
          </Panel>
        </div>
      </main>

      <footer className="foot">
        <span className="mono">SIH26104 · Prevention of Voice-Cloning &amp; Impersonation Attacks</span>
        <span className="mono foot-dim">
          Dashboard (E) + reasoning (F) + call simulator (G) · speaks the task C WebSocket contract
        </span>
      </footer>
    </div>
  );
}
