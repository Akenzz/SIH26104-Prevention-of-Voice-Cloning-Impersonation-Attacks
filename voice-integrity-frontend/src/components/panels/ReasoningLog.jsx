import { useEffect, useRef, useState } from "react";

// Task F — streams the grounded narration lines. The newest local line "types"
// in for a live-reasoning feel; LLM lines stream token-by-token with a caret.
// Every line is a paraphrase of the detector's own numbers (see lib/narrator.js
// and the backend /narrate allowlist). This is a narration convenience, NOT
// explainable AI — the header says so and no evidence is inferred beyond the
// scores shown.

const KIND_CLASS = {
  collect: "text-zinc-400",
  info: "text-zinc-300",
  warn: "text-amber-400",
  alert: "text-red-400",
  unavailable: "text-zinc-500 italic",
  verdict: "text-zinc-100 font-medium",
};

export default function ReasoningLog({ reasoning }) {
  const listRef = useRef(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [reasoning]);

  return (
    <div className="flex flex-col">
      <p className="text-xs text-zinc-500 mb-3 leading-relaxed">
        Plain-language narration of the detector&apos;s own numbers — a readability
        aid, not explainable AI. No evidence is inferred beyond the scores shown.
      </p>
      <ul
        ref={listRef}
        className="space-y-2 max-h-72 overflow-y-auto pr-1 font-mono text-sm"
      >
        {reasoning.length === 0 && (
          <li className="text-zinc-600">
            Narration appears here, grounded only in the detector&apos;s own numbers.
          </li>
        )}
        {reasoning.map((line, i) => {
          const isLast = i === reasoning.length - 1;
          const cls = KIND_CLASS[line.kind] || "text-zinc-300";
          return (
            <li key={line.id} className={`flex gap-2 ${cls}`}>
              <span className="text-zinc-600 select-none" aria-hidden="true">
                ▹
              </span>
              {line.streaming ? (
                <span>
                  {line.text}
                  <span
                    className="inline-block w-1.5 h-4 bg-current opacity-70 animate-pulse ml-0.5 align-middle"
                    aria-hidden="true"
                  />
                </span>
              ) : isLast ? (
                <TypeLine key={line.text} text={line.text} />
              ) : (
                <span>{line.text}</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// The caller passes key={text}, so a new line mounts a fresh TypeLine and the
// typed length resets through useState — no setState in the effect body.
function TypeLine({ text }) {
  const [reduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  const [n, setN] = useState(() => (reduced ? text.length : 0));
  useEffect(() => {
    if (reduced) return;
    let i = 0;
    const id = setInterval(() => {
      i += 2;
      setN(i);
      if (i >= text.length) clearInterval(id);
    }, 14);
    return () => clearInterval(id);
  }, [reduced, text.length]);
  return <span>{text.slice(0, n)}</span>;
}
