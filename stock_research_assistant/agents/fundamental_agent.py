"""
agents/fundamental_agent.py
────────────────────────────
Fundamental Agent – ReAct agent with iteration limit.

Responsibilities:
  - Fetch stock price and financial ratios via MCP tools (get_price, get_financials).
  - Look up financial concept definitions via RAG (financial_knowledge_base).
  - Produce a structured fundamental analysis section with interpreted metrics.

This agent is a mandatory ReAct agent (create_react_agent) with MAX_ITERATIONS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.tools import get_fundamental_agent_tools
from graph.state import ResearchState

# ── Prompt ─────────────────────────────────────────────────────────────────────

FUNDAMENTAL_SYSTEM_PROMPT = """You are a Fundamental Analysis Expert for Indian equities.
Your job is to retrieve and INTERPRET financial data for the given stock(s).

RULES:
1. You MUST NOT say "buy", "sell", "purchase", "short" or give any price targets.
2. Always state the date of the data you are using.
3. Use the get_financials tool to fetch ratios (P/E, P/B, ROE, margins, D/E).
4. Use the get_price tool to fetch current price and 52-week range.
5. Use the financial_knowledge_base tool to look up what ratios MEAN and how to interpret them.
6. Interpret each ratio in the context of the company's sector.
7. Present data in clear sections: Valuation, Profitability, Debt & Liquidity, Growth.
8. End with 2–3 open questions an investor should further investigate.

Output a structured markdown section ready for inclusion in a research note.
"""


def fundamental_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Fundamental Agent (ReAct with iteration limit).
    """
    tickers = state.get("tickers", [])
    if not tickers:
        return {
            "fundamental_data": "No tickers identified – skipping fundamental analysis.",
            "agent_messages": ["[Fundamental Agent] No tickers to analyse."],
            "iteration_counts": {**state.get("iteration_counts", {}), "fundamental": 0},
        }

    # Build tools
    tools = get_fundamental_agent_tools()

    # LLM with temperature=0 for factual accuracy
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )

    # Create ReAct agent (LangGraph 1.x prebuilt)
    # recursion_limit enforces the mandatory iteration limit guardrail
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=FUNDAMENTAL_SYSTEM_PROMPT,
    )

    # Compose the task prompt
    ticker_str = ", ".join(tickers)
    task = (
        f"Perform fundamental analysis for: {ticker_str}.\n"
        f"Original user query: {state['user_query']}"
    )

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": config.MAX_ITERATIONS * 2},  # ← mandatory iteration limit
        )
        messages = result.get("messages", [])
        if messages:
            last_content = messages[-1].content
            if isinstance(last_content, str):
                output = last_content
            elif isinstance(last_content, list):
                output = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in last_content)
            else:
                output = str(last_content)
        else:
            output = "No output produced."
        # Count tool call steps
        steps = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)
    except Exception as exc:
        output = f"Fundamental analysis encountered an error: {exc}"
        steps = 0

    log_msg = f"[Fundamental Agent] Completed in {steps} ReAct steps for {ticker_str}."
    print(log_msg)

    return {
        "fundamental_data": output,
        "agent_messages": [log_msg],
        "iteration_counts": {"fundamental": steps},
    }
