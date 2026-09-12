// Task F — remote (LLM) narration path. Streams a one-line rephrasing of one
// window's numbers from the backend /narrate SSE endpoint, calling `onToken`
import { API_CONFIG } from '../config';
// with the accumulated text so far on each delta. Resolves with the final text.
//
// Throws on any non-OK response (e.g. 503 when no GROQ_API_KEY), a missing body,
// a mid-stream error event, or a timeout — so the caller can fall back to the
// local template narrator (lib/narrator.js). The LLM only ever sees the grounded
// fields the backend allowlists, so this adds no new hallucination surface.

export async function narrateRemote(fields, onToken, { timeoutMs = 12000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(API_CONFIG.BACKEND_URL + "/narrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields),
      signal: ctrl.signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(`narrate ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let acc = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line.
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);

        let evt = "message";
        let data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) evt = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }

        if (evt === "error") throw new Error("narrate upstream error");
        if (evt === "done") return acc;
        if (!data) continue;
        try {
          const parsed = JSON.parse(data);
          if (parsed.delta) {
            acc += parsed.delta;
            onToken(acc);
          }
        } catch {
          // ignore keepalives / non-JSON payloads
        }
      }
    }
    return acc;
  } finally {
    clearTimeout(timer);
  }
}
