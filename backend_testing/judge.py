"""
Judge agent: Granite‑7B‑Lab via Ollama

Granite runs locally through Ollama’s HTTP API (default: http://localhost:11434).

Env vars
--------
JUDGE_MODEL       granite-7b-lab   
JUDGE_MAX_RETRY   3               
"""

from __future__ import annotations

import os, requests, json, time
import pydantic as pd

OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("JUDGE_MODEL", "granite3.3:2b")

#MODEL         = os.getenv("JUDGE_MODEL", "granite-7b-lab")
MAX_RETRY     = int(os.getenv("JUDGE_MAX_RETRY", 3))

# ---------- Pydantic verdict schema ---------------------------------------

class Verdict(pd.BaseModel):
    score: int = pd.Field(ge=0, le=5)
    reasoning: str

# ---------- Core helper ----------------------------------------------------


def _ollama_chat(model: str, prompt: str) -> str:
    """
    Compatible with Ollama 0.9.x: use /api/generate with streaming.
    We stream because 'stream:false' can still block while large models load.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    resp = requests.post(url, json=payload, stream=True, timeout=600)
    resp.raise_for_status()

    parts = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line.decode())
        except json.JSONDecodeError:
            continue
        if "response" in data:
            parts.append(data["response"])
        if data.get("done"):
            break
    return "".join(parts)



# ---------- Public judge function -----------------------------------------

def judge(query: str, answer: str, reference: str | None = None) -> Verdict:
    prompt = (
        "You are an expert telecom support evaluator.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{answer}\n"
    )
    if reference:
        prompt += f"\nREFERENCE (ground truth):\n{reference}\n"
    prompt += (
        "\nRate the ANSWER on factual correctness & helpfulness.\n"
        "Return only valid JSON: {\"score\": 0‑5 integer, \"reasoning\": \"...\"}"
    )

    for attempt in range(1, MAX_RETRY + 1):
        try:
            raw = _ollama_chat(MODEL, prompt).strip()
            return Verdict.model_validate_json(raw)
        except (requests.HTTPError, pd.ValidationError) as e:
            if attempt == MAX_RETRY:
                raise RuntimeError(f"Judge failed after {MAX_RETRY} tries") from e
            time.sleep(1)

# quick CLI smoke‑test
if __name__ == "__main__":
    q = input("Q: ")
    a = input("A: ")
    print(judge(q, a))
