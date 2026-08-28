import {
  Area,
  AreaChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BANDS, pct, riskMeta } from "@/lib/contract.js";

// Strip-chart recorder: smoothed spoof probability over the rolling window, with
// the three policy bands shaded behind the trace. The trace is the measured
// signal (phosphor cyan); the shaded zones carry the risk semantics. Null
// probabilities (unavailable windows) break the line rather than reading as 0.

function TooltipCard({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  const meta = riskMeta(d.state);
  return (
    <div className="chart-tip">
      <span className="mono chart-tip-win">window {d.window}</span>
      <span className="chart-tip-p mono" style={{ color: meta.color }}>
        {d.p == null ? "unavailable" : pct(d.p, 1)}
      </span>
      <span className="chart-tip-state" style={{ color: meta.color }}>
        {meta.label}
      </span>
    </div>
  );
}

export default function TimelineChart({ series }) {
  const last = series.length ? series[series.length - 1] : null;

  const renderDot = (props) => {
    const { cx, cy, index, payload } = props;
    if (index !== series.length - 1 || payload.p == null || cx == null) {
      return <g key={props.key} />;
    }
    const meta = riskMeta(payload.state);
    return (
      <g key={props.key}>
        <circle cx={cx} cy={cy} r={6} fill={meta.color} opacity={0.25}>
          <animate attributeName="r" values="5;9;5" dur="1.4s" repeatCount="indefinite" />
        </circle>
        <circle cx={cx} cy={cy} r={3.5} fill={meta.color} stroke="#0a0d14" strokeWidth={1.5} />
      </g>
    );
  };

  return (
    <div className="chart-wrap">
      {series.length === 0 && (
        <div className="chart-empty mono">waiting for the first 4&nbsp;s window…</div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={series} margin={{ top: 8, right: 10, bottom: 4, left: -18 }}>
          <defs>
            <linearGradient id="traceFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--signal)" stopOpacity={0.28} />
              <stop offset="100%" stopColor="var(--signal)" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <ReferenceArea y1={0} y2={BANDS.low_max} fill="var(--r-low)" fillOpacity={0.06} />
          <ReferenceArea y1={BANDS.low_max} y2={BANDS.uncertain_max} fill="var(--r-uncertain)" fillOpacity={0.06} />
          <ReferenceArea y1={BANDS.uncertain_max} y2={1} fill="var(--r-high)" fillOpacity={0.07} />

          <ReferenceLine y={BANDS.low_max} stroke="var(--r-uncertain)" strokeDasharray="3 4" strokeOpacity={0.5} />
          <ReferenceLine y={BANDS.uncertain_max} stroke="var(--r-high)" strokeDasharray="3 4" strokeOpacity={0.5} />

          <XAxis dataKey="idx" hide />
          <YAxis
            domain={[0, 1]}
            ticks={[0, BANDS.low_max, BANDS.uncertain_max, 1]}
            tickFormatter={(v) => pct(v)}
            tick={{ fill: "var(--ink-faint)", fontSize: 11, fontFamily: "var(--mono)" }}
            axisLine={false}
            tickLine={false}
            width={54}
          />
          <Tooltip content={<TooltipCard />} cursor={{ stroke: "var(--line)" }} isAnimationActive={false} />

          <Area
            type="monotone"
            dataKey="p"
            stroke="var(--signal)"
            strokeWidth={2}
            fill="url(#traceFill)"
            connectNulls={false}
            isAnimationActive={false}
            dot={renderDot}
            activeDot={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      {last && (
        <div className="chart-legend mono">
          <span><i className="sw" style={{ background: "var(--r-low)" }} /> low &lt;{pct(BANDS.low_max)}</span>
          <span><i className="sw" style={{ background: "var(--r-uncertain)" }} /> uncertain</span>
          <span><i className="sw" style={{ background: "var(--r-high)" }} /> high &ge;{pct(BANDS.uncertain_max)}</span>
        </div>
      )}
    </div>
  );
}
