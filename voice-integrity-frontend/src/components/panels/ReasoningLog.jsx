import { useEffect, useRef, useState } from "react";

// Streams the grounded narration lines. Newest line types out for a live feel;
// every line is a paraphrase of real fields (see lib/narrator.js). This is a
// narration convenience, not explainable AI — the header says so.
export default function ReasoningLog({ reasoning }) {
  const listRef = useRef(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [reasoning]);

  return (
    <div className="log">
      <ul className="log-lines" ref={listRef}>
        {reasoning.length === 0 && (
          <li className="log-empty mono">
            Narration appears here, grounded only in the detector's own numbers.
          </li>
        )}
        {reasoning.map((line, i) => {
          const isLast = i === reasoning.length - 1;
          return (
            <li key={line.id} className={`log-line log-${line.kind}`}>
              <span className="log-mark" aria-hidden="true">▹</span>
              {isLast ? <TypeLine text={line.text} /> : <span>{line.text}</span>}
            </li>
          );
        })}
      </ul>
      <div className="log-caret mono" aria-hidden="true">
        <span className="blink">▍</span>
      </div>
    </div>
  );
}

function TypeLine({ text }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    setN(0);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setN(text.length);
      return;
    }
    let i = 0;
    const id = setInterval(() => {
      i += 2;
      setN(i);
      if (i >= text.length) clearInterval(id);
    }, 14);
    return () => clearInterval(id);
  }, [text]);
  return <span>{text.slice(0, n)}</span>;
}
