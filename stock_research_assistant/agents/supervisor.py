"""
agents/supervisor.py
─────────────────────
Supervisor node – the entry point of the LangGraph.

Responsibilities:
  1. Parse the user query to extract stock tickers and portfolio weights.
  2. Classify the query_type:
       - "conceptual"      → user asks about a concept (P/E, beta, etc.)
       - "fundamental"     → single-stock fundamental analysis requested
       - "sentiment"       → news / sentiment about a company requested
       - "portfolio_risk"  → portfolio holdings and risk analysis requested
       - "full_research"   → comprehensive research note requested
  3. Write tickers, weights and query_type into shared state so the router
     can dispatch to the correct downstream agents.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from graph.state import ResearchState

# ── Prompt ─────────────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a Stock Research Assistant.
Your job is to analyse the user's query and produce a JSON object that plans
the execution.

Output ONLY valid JSON with these keys:
{
  "tickers": ["LIST", "OF", "TICKERS"],          // NSE symbols without .NS suffix; [] if no tickers
  "portfolio_weights": [0.40, 0.30, 0.30],        // weights summing to 1.0; [] if not a portfolio query
  "query_type": "<one of: conceptual | fundamental | sentiment | portfolio_risk | full_research>",
  "reasoning": "<one sentence explaining your classification>"
}

Rules:
- "conceptual"     → the user is asking what a financial term means (P/E, beta, ROE, etc.)
- "fundamental"    → the user wants fundamental data for ONE stock (ratios, financials, business)
- "sentiment"      → the user wants news or recent sentiment about a company
- "portfolio_risk" → the user provides multiple stocks with weights and asks about risk
- "full_research"  → the user asks for a complete research note / summary on one stock
- If uncertain, default to "full_research".

Common Indian tickers: INFY (Infosys), TCS (Tata Consultancy Services),
HDFCBANK (HDFC Bank), RELIANCE (Reliance Industries), TATAMOTORS (Tata Motors),
WIPRO, ICICIBANK, AXISBANK, HINDUNILVR, NESTLEIND, BAJFINANCE, KOTAKBANK.

Example:
Query: "Give me a research summary of Infosys."
Output: {"tickers":["INFY"],"portfolio_weights":[],"query_type":"full_research","reasoning":"User wants a full research note on Infosys."}
"""


def supervisor_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Supervisor.
    Parses the user query and classifies it.
    """
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]

    response = llm.invoke(messages)
    if isinstance(response.content, str):
        raw = response.content.strip()
    elif isinstance(response.content, list):
        # Handle list of content blocks from langchain_google_genai
        raw = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content).strip()
    else:
        raw = str(response.content).strip()

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback
        parsed = {
            "tickers": [],
            "portfolio_weights": [],
            "query_type": "full_research",
            "reasoning": "Could not parse supervisor output; defaulting to full_research.",
        }

    tickers: list[str] = parsed.get("tickers", [])
    weights: list[float] = parsed.get("portfolio_weights", [])
    query_type: str = parsed.get("query_type", "full_research")

    log_msg = (
        f"[Supervisor] Query classified as '{query_type}'. "
        f"Tickers: {tickers}. Weights: {weights}. "
        f"Reason: {parsed.get('reasoning', '')}"
    )
    print(log_msg)

    return {
        "tickers": tickers,
        "portfolio_weights": weights,
        "query_type": query_type,
        "agent_messages": [log_msg],
        "fundamental_data": None,
        "sentiment_data": None,
        "risk_data": None,
        "rag_answer": None,
        "final_report": None,
        "error": None,
        "iteration_counts": {},
    }
