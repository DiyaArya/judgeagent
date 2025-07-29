import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import streamlit as st
from backend_testing.observation import logger
  
from backend_testing.results import Query, QueryAgent
from backend_testing.judge   import judge

# ── page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vi-FAQ • LLM-Judged", page_icon="📱")
st.title("📱 Vi-FAQ Assistant")
st.caption("Ask a question about Vi services and see how a Judge LLM scores the answer.")

LOGFIRE_URL = os.getenv("LOGFIRE_PROJECT_URL", "https://logfire.pydantic.dev/app")

# ── user query input ──────────────────────────────────────────────────────
query_text = st.text_input("🔎 Your question", placeholder="e.g. How do I check my Vi balance?")

if query_text:
    with st.spinner("Retrieving answer…"):
        agent   = QueryAgent()
        query   = Query(query_text=query_text)
        result  = agent.run(query)

    st.markdown("### 💬 Answer")
    st.success(result.answer)

    if result.fallback_needed:
        st.warning("⚠️ No close match in FAQ: fallback (Tavily/OpenAI) was used.")

    # ── judge evaluation ─────────────────────────────────────────────────
    with st.spinner("Evaluating with Judge LLM…"):
        try:
            verdict = judge(query_text, result.answer)
            score, reasoning = verdict.score, verdict.reasoning
        except Exception as e:
            score, reasoning = "N/A", f"Judge error: {e}"

    st.markdown("### 🧠 Judge Agent Evaluation")
    st.info(f"**Score:** {score} / 5")
    st.caption(reasoning)

    # simple CSV log (optional)
    with open("runs.csv", "a", newline="") as f:
        f.write(f"{datetime.now().isoformat()},{query_text},{result.answer!r},{score},{reasoning!r}\n")

