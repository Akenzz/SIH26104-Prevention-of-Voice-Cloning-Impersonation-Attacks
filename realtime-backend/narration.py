"""Task F: optional LLM narration of per-window detector numbers (Groq).

This is a *narration convenience*, not explainable AI. The model is handed only
an allowlisted set of the numeric/state fields the detector actually emitted,
plus a locked system prompt that forbids any invented acoustic evidence or
identity claim. When no GROQ_API_KEY is set, callers fall back to the frontend's
local template narrator, so nothing here is required for the product to work.

`httpx` is imported lazily inside `stream_groq` so this module (and the whole
backend) still imports cleanly when the optional dependency is absent.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Only these fields are ever forwarded to the LLM. Anything else the client
# sends is dropped, so the prompt can never widen beyond real detector output.
ALLOWED_FIELDS = (
    "window_index",
    "risk_state",
    "smoothed_probability",
    "raw_per_expert_scores",
    "audio_quality",
    "recommended_action",
)

SYSTEM_PROMPT = (
    "You narrate a speech-liveness detector for a human operator. "
    "You are given a JSON object with ONLY these numeric/state fields and nothing else. "
    "Write ONE short sentence (max 25 words) that restates only these values in plain language. "
    "Rules: never invent acoustic evidence (no spectral artifacts, prosody, breathing, "
    "formants, jitter, shimmer, harmonics, pitch, or timbre); never claim or deny a person's "
    "identity; do not add numbers that are not present; if a field is missing, do not mention it. "
    "This is a readability aid, not explainable AI."
)


def sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only the allowlisted fields, so the LLM never sees anything else."""
    return {k: payload[k] for k in ALLOWED_FIELDS if k in payload}


async def stream_groq(
    fields: dict[str, Any], *, api_key: str, model: str
) -> AsyncIterator[str]:
    """Yield content deltas from Groq's OpenAI-compatible streaming endpoint.

    Raises on a missing key, a missing httpx install, or any upstream error, so
    the caller can signal the frontend to fall back to local narration.
    """
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    import httpx  # lazy: only the live LLM path needs the dependency

    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 80,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(fields, ensure_ascii=True)},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(15.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", GROQ_URL, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (
                    chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                )
                if delta:
                    yield delta
