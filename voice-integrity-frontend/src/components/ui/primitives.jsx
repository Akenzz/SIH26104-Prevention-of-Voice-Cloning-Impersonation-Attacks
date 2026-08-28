// Small shared building blocks for the console shell.

export function Panel({ eyebrow, title, right, className = "", children, ...rest }) {
  return (
    <section className={`panel ${className}`} {...rest}>
      {(eyebrow || title || right) && (
        <header className="panel-head">
          <div>
            {eyebrow && <span className="panel-eyebrow mono">{eyebrow}</span>}
            {title && <h2 className="panel-title">{title}</h2>}
          </div>
          {right && <div className="panel-right">{right}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Pill({ tone = "neutral", children, title }) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      <span className="pill-dot" />
      {children}
    </span>
  );
}

export function Stat({ label, value, sub, mono = true, tone }) {
  return (
    <div className="stat">
      <span className="stat-label mono">{label}</span>
      <span className={`stat-value ${mono ? "mono" : ""}`} style={tone ? { color: tone } : undefined}>
        {value}
      </span>
      {sub && <span className="stat-sub mono">{sub}</span>}
    </div>
  );
}

export function Segmented({ options, value, onChange, ariaLabel }) {
  return (
    <div className="segmented" role="tablist" aria-label={ariaLabel}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={active}
            className={`seg ${active ? "seg-on" : ""}`}
            onClick={() => onChange(o.value)}
            disabled={o.disabled}
            title={o.title}
          >
            {o.icon}
            <span>{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}
