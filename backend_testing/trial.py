"""

• QueryAgent  (retrieves from chroma_db)
• Fallback    (search_fallback or idk_fallback inside QueryAgent)
• Judge       (Granite‑7B via Ollama) –> score + reasoning
• Logging     (CSV via logger.py  +  Logfire via observation.py)

Run with:
    source .venv/bin/activate            
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager

from backend_testing.results import Query, QueryAgent
from backend_testing.observation import logger, span           # Logfire shortcuts
from backend_testing.judge import judge                       # Granite‑powered judge

# ---------- Optional CSV helper -------------------------------------------
try:
    from backend_testing.logger import log_row                # writes runs.csv
except ModuleNotFoundError:
    log_row = None



def main() -> None:
    agent = QueryAgent()
    print("⚡  Vi‑FAQ bot  (press Enter on a blank line to quit)\n")

    while True:
        query_text = input("enter your query: ").strip()
        if not query_text:
            print("Good‑bye!")
            break

        with span("query_cycle", query=query_text):
            t0       = time.time()
            result   = agent.run(Query(query_text=query_text))
            elapsed  = int((time.time() - t0) * 1000)

            # --- unpack result object -------------------------------------
            answer   = getattr(result, "answer", str(result))
            fallback = getattr(result, "fallback_needed", False)
            ids      = getattr(result, "ids", [])

            # --- judge evaluation -----------------------------------------
            try:
                verdict = judge(query_text, answer)
                score, reason = verdict.score, verdict.reasoning
            except Exception as e:        # Judge failure should not crash the bot
                score, reason = None, f"Judge error: {e}"

            # --- console output -------------------------------------------
            print(f"\nAnswer  ➜ {answer}")
            if score is not None:
                print(f"Judge   ➜ {score}/5 · {reason}")
            print()

            # --- CSV logging ----------------------------------------------
            if log_row:
                log_row(
                    ts=time.time(),
                    query=query_text,
                    answer=answer,
                    fallback=fallback,
                    latency_ms=elapsed,
                    score=score or "",
                    reason=reason,
                    ids=json.dumps(ids),
                )

            # --- Logfire structured event --------------------------------
            logger.info(
                "faq.answer.done",
                query=query_text,
                answer=answer,
                fallback=fallback,
                latency_ms=elapsed,
                score=score,
                ids=ids,
            )



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")





















'''from __future__ import annotations

import json, sys, time
from contextlib import contextmanager

from backend_testing.results import Query, QueryAgent
from backend_testing.observation import logger, span          # logs → Logfire

try:
    from backend_testing.judge import judge   # real module
 
# ---- optional CSV logger ----------------------------------
try:
    from backend_testing.logger import log_row
except ModuleNotFoundError:
    log_row = None

# ------------- main driver ---------------------------------
def main() -> None:
    agent = QueryAgent()
    print("⚡  Vi‑FAQ bot  (hit Enter on empty line to quit)\n")

    while True:
        query_text = input("enter your query: ").strip()      # ← prompt you saw
        if not query_text:
            print("Good‑bye!")
            break

        with span("query_cycle", query=query_text):
            t0      = time.time()
            result  = agent.run(Query(query_text=query_text))

            # ① Derive the answer string --------------------
            answer  = getattr(result, "answer", str(result))

            # ② Optional judge ------------------------------
            if judge is not None:
                verdict = judge(query_text, answer)
                score, reasoning = verdict.score, verdict.reasoning
            else:
                score, reasoning = None, ""

            elapsed = int((time.time() - t0) * 1000)

            # ③ Print to console ----------------------------
            print(f"\nAnswer  ➜ {answer}")
            if score is not None:
                print(f"Judge   ➜ {score}/5 · {reasoning}")
            print()  # blank line

            # ④ Log to CSV + Logfire ------------------------
            if log_row:
                log_row(ts=time.time(),
                        query=query_text,
                        answer=answer,
                        fallback=getattr(result, "fallback_needed", False),
                        latency_ms=elapsed,
                        score=score or "",
                        reason=reasoning)

            logger.info(                   # Logfire
                "faq.answer.done",
                query=query_text,
                answer=answer,
                fallback=getattr(result, "fallback_needed", False),
                latency_ms=elapsed,
                score=score,
            )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
'''