import { useRef, useState } from "react";
import { Pill, Segmented } from "@/components/ui/primitives.jsx";
import { IconFile, IconLink, IconMic, IconPlay, IconRefresh, IconStop } from "@/components/ui/icons.jsx";
import { SCENARIOS } from "@/lib/simulator.js";

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function SourcePanel({
  source, setSource,
  scenario, setScenario,
  inputKind, setInputKind,
  backendBase, setBackendBase,
  status, running, error, elapsedSec,
  health, healthError, refreshHealth,
  onStart, onStop,
}) {
  const [file, setFile] = useState(null);
  const fileInput = useRef(null);
  const busy = running;

  const needFile = source === "live" && inputKind === "file";
  const canStart = !needFile || !!file;

  const scenarioMeta = SCENARIOS.find((s) => s.id === scenario);

  const handleStart = () => {
    if (source === "live" && inputKind === "file") onStart(file);
    else onStart();
  };

  return (
    <div className="deck">
      <div className="deck-block">
        <span className="deck-label mono">SOURCE</span>
        <Segmented
          ariaLabel="Signal source"
          value={source}
          onChange={(v) => !busy && setSource(v)}
          options={[
            { value: "sim", label: "Simulation", icon: <IconPlay size={15} /> },
            { value: "live", label: "Live backend", icon: <IconLink size={15} /> },
          ]}
        />
      </div>

      {source === "sim" ? (
        <div className="deck-block">
          <span className="deck-label mono">SCENARIO</span>
          <div className="scenario-list">
            {SCENARIOS.map((s) => (
              <button
                key={s.id}
                className={`scenario ${scenario === s.id ? "scenario-on" : ""}`}
                onClick={() => !busy && setScenario(s.id)}
                disabled={busy}
              >
                <span className="scenario-name">{s.label}</span>
                <span className="scenario-blurb">{s.blurb}</span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          <div className="deck-block">
            <span className="deck-label mono">INPUT</span>
            <Segmented
              ariaLabel="Live input"
              value={inputKind}
              onChange={(v) => !busy && setInputKind(v)}
              options={[
                { value: "mic", label: "Microphone", icon: <IconMic size={15} /> },
                { value: "file", label: "Audio file", icon: <IconFile size={15} /> },
              ]}
            />
          </div>

          {inputKind === "file" && (
            <div className="deck-block">
              <span className="deck-label mono">FILE</span>
              <div className="file-pick">
                <button className="btn-ghost" onClick={() => fileInput.current?.click()} disabled={busy}>
                  <IconFile size={15} /> {file ? "Change" : "Choose audio"}
                </button>
                <span className="file-name mono">{file ? file.name : "no file selected"}</span>
                <input
                  ref={fileInput}
                  type="file"
                  accept="audio/*,.wav,.flac,.ogg,.mp3"
                  hidden
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
              </div>
            </div>
          )}

          <div className="deck-block">
            <span className="deck-label mono">BACKEND</span>
            <div className="backend-row">
              <input
                className="field mono"
                placeholder="same-origin (/ws proxy)"
                value={backendBase}
                onChange={(e) => setBackendBase(e.target.value)}
                disabled={busy}
                spellCheck={false}
              />
              <button className="btn-ghost" onClick={refreshHealth} title="Check /health">
                <IconRefresh size={15} /> Check
              </button>
            </div>
            <div className="health">
              {health ? (
                <>
                  <Pill tone="ok">online</Pill>
                  <span className="mono health-bit">experts: {(health.experts || []).join(", ") || "—"}</span>
                  <span className="mono health-bit">{health.fusion_mode}</span>
                  <span className="mono health-bit">win {health.window_sec}s / hop {health.hop_sec}s</span>
                </>
              ) : healthError ? (
                <Pill tone="bad" title={healthError}>backend unreachable</Pill>
              ) : (
                <span className="mono muted">health not checked</span>
              )}
            </div>
          </div>
        </>
      )}

      <div className="deck-actions">
        <button
          className={`btn-primary ${running ? "btn-stop" : ""}`}
          onClick={running ? onStop : handleStart}
          disabled={!running && !canStart}
        >
          {running ? <IconStop size={18} /> : <IconPlay size={18} />}
          {running ? "Stop" : source === "sim" ? "Run simulation" : inputKind === "file" ? "Analyze file" : "Start monitor"}
        </button>
        <div className="deck-status">
          <StatusDot status={status} />
          <span className="mono">
            {status === "running" && `live · ${fmtTime(elapsedSec)}`}
            {status === "connecting" && "connecting…"}
            {status === "idle" && "idle"}
            {status === "stopped" && "stopped"}
            {status === "error" && "error"}
          </span>
        </div>
      </div>

      {source === "sim" && scenarioMeta && !running && (
        <p className="deck-hint">{scenarioMeta.blurb}</p>
      )}
      {needFile && !file && !running && (
        <p className="deck-hint">Choose an audio file to stream through the backend.</p>
      )}
      {error && <p className="deck-error">{error}</p>}
    </div>
  );
}

function StatusDot({ status }) {
  const tone =
    status === "running" ? "ok" : status === "error" ? "bad" : status === "connecting" ? "warn" : "neutral";
  return <span className={`sdot sdot-${tone}`} />;
}
