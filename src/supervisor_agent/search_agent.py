"""Self-ask-with-search style agent using Google Serper."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm import default_llm


SELF_ASK_PROMPT = """\
You can ask intermediate questions to search the web.
If you need to search, respond with: SEARCH: <query>
When you have the final answer, respond with: FINAL: <answer>
Keep reasoning brief and use at most 3 searches.
"""


@lru_cache(maxsize=1)
def _search_client() -> GoogleSerperAPIWrapper:
    return GoogleSerperAPIWrapper()


def ask_with_search(question: str) -> str:
    """Run a lightweight self-ask-with-search loop."""
    llm = default_llm(temperature=0.7)
    messages: List = [
        SystemMessage(content=SELF_ASK_PROMPT),
        HumanMessage(content=question),
    ]
    search = _search_client()

    for _ in range(4):
        reply = llm.invoke(messages).content
        if not isinstance(reply, str):
            return str(reply)
        upper = reply.strip()
        if upper.lower().startswith("final:"):
            return reply.split(":", 1)[1].strip()
        if upper.lower().startswith("search:"):
            query = reply.split(":", 1)[1].strip()
            results = search.run(query)
            messages.append(AIMessage(content=reply))
            messages.append(HumanMessage(content=f"Search results:\n{results}"))
            continue
        # fallback: return model output if it didn't follow protocol
        return reply

    return reply
