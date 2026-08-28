const variants = {
  default: 'bg-zinc-800 text-zinc-100 border-zinc-700',
  success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  warning: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  danger: 'bg-red-500/10 text-red-400 border-red-500/20',
  collecting: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
};

export function Badge({ children, variant = 'default', className = '' }) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${variants[variant]} ${className}`}
    >
      {children}
    </span>
  );
}

export function RiskBadge({ state }) {
  const map = {
    low: { v: 'success', l: 'Low Risk (Bonafide)' },
    uncertain: { v: 'warning', l: 'Uncertain' },
    high: { v: 'danger', l: 'High Risk (Spoof)' },
    collecting: { v: 'collecting', l: 'Collecting...' },
    unavailable: { v: 'default', l: 'Unavailable' }
  };
  const s = map[state?.toLowerCase()] || map.unavailable;
  return <Badge variant={s.v}>{s.l}</Badge>;
}
