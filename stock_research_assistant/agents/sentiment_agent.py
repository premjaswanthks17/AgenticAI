"""
agents/sentiment_agent.py
──────────────────────────
Sentiment Agent – ReAct agent with iteration limit.

Responsibilities:
  - Use Tavily to search for recent news about the company (last 30 days).
  - Classify overall sentiment as Positive / Neutral / Negative with evidence.
  - Identify key themes: earnings beats/misses, regulatory actions, management changes,
    product launches, macro headwinds, analyst upgrades/downgrades.
  - Use RAG to provide context on how news events typically affect valuation.

This is a mandatory ReAct agent with MAX_ITERATIONS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from agents.tools import get_sentiment_agent_tools
from graph.state import ResearchState

# ── Prompt ─────────────────────────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = """You are a News Sentiment Analyst for Indian equities.
Your job is to gather RECENT news about the given company and produce a balanced
sentiment analysis with evidence.

RULES:
1. NEVER say "buy", "sell" or give price targets.
2. Always state the date range of the news you searched.
3. Use the tavily_news_search tool to find recent news (search for 'COMPANY_NAME recent news 2024').
4. Search for at least 2–3 different angles: earnings, regulatory, sector trends.
5. Use the financial_knowledge_base if you need context on financial events.
6. Classify sentiment as one of: Strongly Positive / Mildly Positive / Neutral /
   Mildly Negative / Strongly Negative.
7. Provide 3–5 specific evidence points (with source URLs where available).
8. Identify key themes: earnings performance, management changes, regulatory news,
   competitive dynamics, macro factors.

Output Format (Markdown):
## News Sentiment Analysis: [Company Name]
**Overall Sentiment:** [label]
**Data as of:** [date]

### Key Evidence
- [Evidence point 1 with source]
...

### Key Themes
- Theme 1: ...
...

### Sentiment Summary
[2–3 sentence balanced summary]
"""


def sentiment_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Sentiment Agent (ReAct with iteration limit).
    """
    tickers = state.get("tickers", [])
    if not tickers:
        return {
            "sentiment_data": "No tickers identified – skipping sentiment analysis.",
            "agent_messages": ["[Sentiment Agent] No tickers to analyse."],
            "iteration_counts": {**state.get("iteration_counts", {}), "sentiment": 0},
        }

    # Build tools
    tools = get_sentiment_agent_tools()

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY,
    )

    # Create ReAct agent (LangGraph 1.x prebuilt)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SENTIMENT_SYSTEM_PROMPT,
    )

    # Map tickers to full company names for better news search queries
    ticker_name_map = {
        "INFY": "Infosys", "TCS": "Tata Consultancy Services",
        "HDFCBANK": "HDFC Bank", "RELIANCE": "Reliance Industries",
        "TATAMOTORS": "Tata Motors", "WIPRO": "Wipro",
        "ICICIBANK": "ICICI Bank", "AXISBANK": "Axis Bank",
        "HINDUNILVR": "Hindustan Unilever", "NESTLEIND": "Nestle India",
        "BAJFINANCE": "Bajaj Finance", "KOTAKBANK": "Kotak Mahindra Bank",
        "ONGC": "ONGC", "NTPC": "NTPC",
        "SUNPHARMA": "Sun Pharmaceutical", "DRREDDY": "Dr Reddy's Laboratories",
    }
    company_names = [ticker_name_map.get(t.upper(), t) for t in tickers]
    ticker_str = ", ".join(tickers)
    company_str = ", ".join(company_names)


    task = (
        f"Analyse news sentiment for: {company_str} (tickers: {ticker_str}).\n"
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
        output = f"Sentiment analysis encountered an error: {exc}"
        steps = 0

    log_msg = f"[Sentiment Agent] Completed in {steps} ReAct steps for {ticker_str}."
    print(log_msg)

    return {
        "sentiment_data": output,
        "agent_messages": [log_msg],
        "iteration_counts": {"sentiment": steps},
    }
