// Inline stroke icons (currentColor) so risk hues and the phosphor accent flow
// straight from CSS. Hand-rolled to avoid any icon-package version drift.

function Svg({ size = 24, children, ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconCollecting = (p) => (
  <Svg {...p}>
    <path d="M12 3v2M12 19v2M3 12h2M19 12h2" opacity="0.5" />
    <circle cx="12" cy="12" r="3" />
    <path d="M6.5 6.5a7 7 0 0 0 0 11M17.5 6.5a7 7 0 0 1 0 11" opacity="0.7" />
  </Svg>
);

export const IconLow = (p) => (
  <Svg {...p}>
    <path d="M12 3l7 3v5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6l7-3z" />
    <path d="M9 12l2 2 4-4" />
  </Svg>
);

export const IconUncertain = (p) => (
  <Svg {...p}>
    <path d="M10.3 4.3l-6 6a2.4 2.4 0 0 0 0 3.4l6 6a2.4 2.4 0 0 0 3.4 0l6-6a2.4 2.4 0 0 0 0-3.4l-6-6a2.4 2.4 0 0 0-3.4 0z" />
    <path d="M12 9.2c1.1 0 1.9.7 1.9 1.7 0 1.5-1.9 1.3-1.9 3M12 15.6h.01" />
  </Svg>
);

export const IconHigh = (p) => (
  <Svg {...p}>
    <path d="M12 4l9 16H3l9-16z" />
    <path d="M12 10v4M12 17h.01" />
  </Svg>
);

export const IconUnavailable = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M6 6l12 12" />
  </Svg>
);

export const IconMic = (p) => (
  <Svg {...p}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </Svg>
);

export const IconFile = (p) => (
  <Svg {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z" />
    <path d="M14 3v5h5" />
    <path d="M9 13.5c.7-1 1.6-1 2.2 0s1.5 1 2.2 0 1.6-1 2.2 0" opacity="0.8" />
  </Svg>
);

export const IconWave = (p) => (
  <Svg {...p}>
    <path d="M3 12h2l2-6 3 14 3-18 3 12 2-4h3" />
  </Svg>
);

export const IconPlay = (p) => (
  <Svg {...p}>
    <path d="M7 5l12 7-12 7V5z" fill="currentColor" stroke="none" />
  </Svg>
);

export const IconStop = (p) => (
  <Svg {...p}>
    <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />
  </Svg>
);

export const IconLink = (p) => (
  <Svg {...p}>
    <path d="M9 15l6-6" />
    <path d="M11 6l1-1a4 4 0 0 1 6 6l-1 1M13 18l-1 1a4 4 0 0 1-6-6l1-1" />
  </Svg>
);

export const IconActivity = (p) => (
  <Svg {...p}>
    <path d="M3 12h4l3 8 4-16 3 8h4" />
  </Svg>
);

export const IconInfo = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8h.01" />
  </Svg>
);

export const IconShield = (p) => (
  <Svg {...p}>
    <path d="M12 3l7 3v5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6l7-3z" />
  </Svg>
);

export const IconRefresh = (p) => (
  <Svg {...p}>
    <path d="M4 12a8 8 0 0 1 14-5.3L21 9M20 12a8 8 0 0 1-14 5.3L3 15" />
    <path d="M21 4v5h-5M3 20v-5h5" />
  </Svg>
);

const RISK_ICON = {
  collecting: IconCollecting,
  low: IconLow,
  uncertain: IconUncertain,
  high: IconHigh,
  unavailable: IconUnavailable,
};

export function RiskGlyph({ state, size = 24, ...rest }) {
  const C = RISK_ICON[state] || IconUnavailable;
  return <C size={size} {...rest} />;
}
