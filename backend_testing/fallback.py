import os
from typing import Optional

import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


TAVILY_KEY   = os.getenv("TAVILY_API_KEY")        
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo-mini")

client = OpenAI(api_key=OPENAI_KEY)


def idk_fallback(_query: str) -> str:
    return "Sorry, I don’t have that information right now."


def search_fallback(query: str) -> str:
    "Tavily → JSON → top snippet → summarise with GPT."
    if not TAVILY_KEY:
        return idk_fallback(query)

    resp = requests.get(
        "https://api.tavily.com/search",
        params={"api_key": TAVILY_KEY, "query": query, "num_results": 5},
        timeout=10,
    )
    results = resp.json()["results"]
    blob = "\n".join(r["content"] for r in results)

    prompt = (
        "Answer the user's question ONLY with information from the text "
        "below. Be concise.\n\nTEXT:\n"
        + blob
        + f"\n\nQUESTION: {query}\nANSWER:"
    )
    cmp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return cmp.choices[0].message.content.strip()
