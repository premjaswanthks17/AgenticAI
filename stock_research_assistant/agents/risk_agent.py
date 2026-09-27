"""
agents/risk_agent.py
─────────────────────
Risk Agent – ReAct agent with iteration limit.

Responsibilities:
  - Call the calc_portfolio_risk MCP tool with the user's holdings and weights.
  - Interpret portfolio volatility, beta, Sharpe ratio and HHI concentration.
  - Use RAG to explain what these metrics mean to a retail investor.
  - Highlight concentration risks and suggest diversification considerations
    (without giving specific buy/sell recommendations).

This is a mandatory ReAct agent with MAX_ITERATIONS.
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
from agents.tools import get_risk_agent_tools
from graph.state import ResearchState

# ── Prompt ─────────────────────────────────────────────────────────────────────

RISK_SYSTEM_PROMPT = """You are a Portfolio Risk Analyst specialising in Indian equities.
Your job is to calculate and EXPLAIN portfolio risk metrics in plain language
for retail investors.

RULES:
1. NEVER say "buy", "sell" or give price targets.
2. Use the calc_portfolio_risk tool to compute metrics.
3. Use the financial_knowledge_base tool to look up what volatility, beta, Sharpe ratio,
   and HHI mean, and how to interpret them.
4. Explain every metric in simple, jargon-free language.
5. Comment on concentration risk (HHI) and sector overlap.
6. Always state the time period used for the calculation and today's date.
7. Offer educational context (e.g., "A beta above 1 means the portfolio moves
   more than the market on average – this is neither good nor bad, but means
   higher swings in both directions.").

Output Format (Markdown):
## Portfolio Risk Analysis
**Holdings:** [list]
**Weights:** [list]
**Analysis Date:** [date]

### Risk Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
...

### Concentration Analysis
...

### Key Risk Considerations
- ...

### Suggested Open Questions for Further Research
- ...
"""


def risk_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Risk Agent (ReAct with iteration limit).
    """
    tickers = state.get("tickers", [])
    weights = state.get("portfolio_weights", [])

    # If no portfolio weights provided, treat as equal-weight
    if tickers and not weights:
        weights = [round(1.0 / len(tickers), 4)] * len(tickers)

    if not tickers:
        return {
            "risk_data": "No portfolio holdings identified – skipping risk analysis.",
            "agent_messages": ["[Risk Agent] No holdings to analyse."],
            "iteration_counts": {**state.get("iteration_counts", {}), "risk": 0},
        }

    # Build tools
    tools = get_risk_agent_tools()

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )

    # Create ReAct agent (LangGraph 1.x prebuilt)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RISK_SYSTEM_PROMPT,
    )

    # Format portfolio for the task
    portfolio_input = json.dumps({"holdings": tickers, "weights": weights})
    ticker_str = ", ".join(
        f"{t} ({w * 100:.1f}%)" for t, w in zip(tickers, weights)
    )

    task = (
        f"Portfolio to analyse:\n{portfolio_input}\n\n"
        f"Human-readable: {ticker_str}\n\n"
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
        steps = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)
    except Exception as exc:
        output = f"Risk analysis encountered an error: {exc}"
        steps = 0

    log_msg = f"[Risk Agent] Completed in {steps} ReAct steps for portfolio: {ticker_str}."
    print(log_msg)

    return {
        "risk_data": output,
        "agent_messages": [log_msg],
        "iteration_counts": {"risk": steps},
    }
