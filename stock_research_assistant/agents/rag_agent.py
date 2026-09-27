"""
agents/rag_agent.py
────────────────────
RAG Agent – handles purely conceptual / educational questions.

When the Supervisor classifies query_type = "conceptual", traffic is routed
here instead of the fundamental/risk/sentiment agents.

The RAG agent:
  1. Queries the knowledge base for relevant passages.
  2. Uses the LLM to synthesise a clear, educational answer.
  3. Cites sources from the knowledge base.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from rag.knowledge_base import get_retriever
from graph.state import ResearchState

RAG_SYSTEM_PROMPT = """You are an educational financial assistant.
A user has asked a conceptual question about finance, investing or markets.
You have been provided with relevant passages from a trusted knowledge base.

RULES:
1. Answer clearly and educationally – assume the user is a beginner.
2. Cite the source document for every key claim (use [Source: filename]).
3. NEVER say "buy", "sell" or give investment recommendations.
4. If the knowledge base passages don't fully answer the question, say so honestly.
5. End with: "Source: [list of documents used]"
"""


def rag_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: RAG Agent for conceptual questions.
    Does NOT use a ReAct loop – a single retrieval + LLM synthesis is sufficient.
    """
    query = state["user_query"]

    # Retrieve relevant chunks
    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        rag_context = "No relevant passages found in the knowledge base."
    else:
        rag_context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'knowledge base')}]\n{doc.page_content}"
            for doc in docs
        )

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question: {query}\n\n"
                f"Relevant knowledge base passages:\n\n{rag_context}"
            )
        ),
    ]

    response = llm.invoke(messages)
    if isinstance(response.content, str):
        answer = response.content
    elif isinstance(response.content, list):
        answer = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content)
    else:
        answer = str(response.content)

    log_msg = f"[RAG Agent] Answered conceptual question using {len(docs)} retrieved passages."
    print(log_msg)

    return {
        "rag_answer": answer,
        "agent_messages": [log_msg],
    }
