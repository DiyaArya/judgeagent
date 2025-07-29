"""
results.py
~~~~~~~~~~
• QueryAgent  — retrieves from ./chroma_db/vi_faq
• Result dataclass — typed wrapper (answer / fallback_needed / ids)
• Optional Tavily + Mistral helpers (only used when handle() is called)
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
import os, requests
from dataclasses import dataclass
from typing import Any, List, Optional

# ── third‑party ───────────────────────────────────────────────────────────
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pydantic import BaseModel
from tavily import TavilyClient

# ── local ─────────────────────────────────────────────────────────────────
from backend_testing.fallback import search_fallback, idk_fallback

# ──────────────────────────────────────────────────────────────────────────
CHROMA_PATH   = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION    = "vi_faq"                         # matches the seeded name
EMBEDDING_FN  = DefaultEmbeddingFunction()

# ── Result wrapper --------------------------------------------------------


@dataclass
class Result:
    answer: str
    fallback_needed: bool
    ids: List[Any]            # chroma doc IDs (or empty list)


# ── Pydantic DTOs for higher‑level agents --------------------------------


class Query(BaseModel):
    query_text: str


class TavilyResult(BaseModel):
    source: str
    link: Optional[str] = None


# ── QueryAgent: retrieval + confidence test ------------------------------


class QueryAgent:
    """Lightweight wrapper around a Chroma collection."""

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_or_create_collection(
            name=COLLECTION, embedding_function=EMBEDDING_FN
        )

    def run(self, query: Query) -> Result:
        res = self.collection.query(
            query_texts=[query.query_text], n_results=1
        )

        answer    = res["documents"][0][0]
        distance  = res["distances"][0][0]
        ids       = res["ids"][0]

        # Threshold > 1.0 → low similarity → use fallback
        if distance > 1:
            return Result(
                answer=search_fallback(query.query_text),   # or idk_fallback(...)
                fallback_needed=True,
                ids=[],
            )

        return Result(answer=answer, fallback_needed=False, ids=ids)


# ── Optional helpers for richer fallback flows ---------------------------


class Tavily:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key)

    def run(self, query: Query) -> TavilyResult:
        try:
            response = self.client.search(query=query.query_text)
            results  = response.get("results", response)

            if isinstance(results, list) and results:
                first = results[0]
                return TavilyResult(source=first["content"], link=first.get("url"))

            return TavilyResult(source="No useful web result found.")
        except Exception as e:
            return TavilyResult(source=f"[Tavily Error] {e}")


class MistralLLM:
    """Local Ollama mistral → simple one‑shot rewrite."""

    def __init__(self, model: str = "mistral"):
        self.model = model

    def run(self, prompt: str) -> str:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return f"[LLM Error] {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"[LLM Connection Failed] {e}"


# ── High‑level chatbot (for ad‑hoc use, not imported by trial.py) ---------


class ChatBot:
    """Simple coordinator for manual testing."""

    def __init__(self, tavily_key: str):
        self.query_agent = QueryAgent()
        self.tavily      = Tavily(tavily_key)
        self.mistral     = MistralLLM()

    def handle(self, user_query: str) -> None:
        q      = Query(query_text=user_query)
        result = self.query_agent.run(q)

        if result.fallback_needed:
            web = self.tavily.run(q)
            print("\n🔍  Fallback via Tavily")
            print(web.source)
            if web.link:
                print("Source:", web.link)
        else:
            prompt = (
                "Rewrite in a friendly style:\n\n"
                f"Q: {user_query}\nA: {result.answer}"
            )
            rewrite = self.mistral.run(prompt)
            print("\n🤖  Mistral rewrite")
            print(rewrite)


# ── Optional CLI test -----------------------------------------------------


if __name__ == "__main__":
    # Ad‑hoc testing without interfering with imports elsewhere
    bot = ChatBot(tavily_key=os.getenv("TAVILY_API_KEY", ""))
    while True:
        try:
            user = input("\nQuery (blank to exit): ").strip()
            if not user:
                break
            bot.handle(user)
        except KeyboardInterrupt:
            break
