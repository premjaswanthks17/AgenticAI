"""
agents/tools.py
───────────────
All LangChain tools used by the sub-agents, grouped into:

  A) MCP Tools (loaded from the custom MCP server via langchain-mcp-adapters)
       - get_price
       - get_financials
       - calc_portfolio_risk

  B) Additional LangChain Tools (mandatory requirement: at least 2)
       1. financial_knowledge_base  – RAG retriever with citations
       2. tavily_news_search        – Tavily web search for current news

Both A and B are exposed through get_all_tools() so agents can bind them.

Note: MCP tools are loaded asynchronously via MultiServerMCPClient; this
module provides a synchronous wrapper for convenience.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from rag.knowledge_base import get_rag_tool

# ── 1. RAG Tool (Additional LangChain Tool #1) ─────────────────────────────────

def get_rag_knowledge_tool() -> Tool:
    """Return the RAG retriever tool with source citations."""
    return get_rag_tool()


# ── 2. Tavily Web Search Tool (Additional LangChain Tool #2) ──────────────────

def get_tavily_tool(max_results: int = 5) -> TavilySearchResults:
    """
    Return a Tavily search tool configured for financial news retrieval.
    This is the second mandatory additional LangChain tool.
    """
    return TavilySearchResults(
        max_results=max_results,
        tavily_api_key=config.TAVILY_API_KEY,
        name="tavily_news_search",
        description=(
            "Search the internet for recent news, earnings announcements, "
            "regulatory actions and market sentiment about a company or stock. "
            "Input: a search query string, e.g. 'Infosys Q3 results 2024' or "
            "'Tata Motors recent news'. Returns URLs, titles and snippets."
        ),
    )


# ── 3. MCP Tools (loaded asynchronously) ─────────────────────────────────────

async def _load_mcp_tools() -> List[BaseTool]:
    """
    Connect to the custom MCP server (stock_mcp_server.py) via stdio
    and load its tools as LangChain BaseTool objects.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        mcp_server_script = config.MCP_SERVER_SCRIPT

        async with MultiServerMCPClient(
            {
                "stock_research": {
                    "command": "python",
                    "args": [mcp_server_script],
                    "transport": "stdio",
                }
            }
        ) as client:
            tools = client.get_tools()
            return tools
    except Exception as e:
        print(f"[MCP] Warning: Could not load MCP tools: {e}")
        print("[MCP] Falling back to direct yfinance function wrappers.")
        return _get_fallback_mcp_tools()


def _get_fallback_mcp_tools() -> List[Tool]:
    """
    Fallback: wrap MCP server functions directly as LangChain Tools
    in case langchain-mcp-adapters is unavailable or the server fails to start.
    """
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
    from stock_mcp_server import get_price, get_financials, calc_portfolio_risk

    def _price_tool(query: str) -> str:
        """Parse 'TICKER [period]' from query string."""
        parts = query.strip().split()
        ticker = parts[0] if parts else "NIFTY50"
        period = parts[1] if len(parts) > 1 else "1mo"
        return get_price(ticker, period)

    def _financials_tool(ticker: str) -> str:
        return get_financials(ticker.strip())

    def _risk_tool(query: str) -> str:
        """
        Parse JSON input: {"holdings": ["TCS","HDFC"], "weights": [0.5, 0.5]}
        or comma-separated: 'TCS:0.4,HDFC:0.3,RELIANCE:0.3'
        """
        try:
            data = json.loads(query)
            return calc_portfolio_risk(data["holdings"], data["weights"])
        except Exception:
            # Try CSV format
            items = [item.strip() for item in query.split(",")]
            holdings, weights = [], []
            for item in items:
                if ":" in item:
                    ticker, w = item.split(":", 1)
                    holdings.append(ticker.strip())
                    weights.append(float(w.strip()))
                else:
                    holdings.append(item)
            if not weights:
                weights = [1 / len(holdings)] * len(holdings)
            return calc_portfolio_risk(holdings, weights)

    return [
        Tool(
            name="get_price",
            func=_price_tool,
            description=(
                "Fetch current and historical price data for a stock. "
                "Input format: 'TICKER period' e.g. 'TCS 3mo' or 'INFY 1y'. "
                "Indian tickers: TCS, INFY, RELIANCE, HDFCBANK, TATAMOTORS. "
                "Returns JSON with OHLCV data, 52-week high/low, and % change."
            ),
        ),
        Tool(
            name="get_financials",
            func=_financials_tool,
            description=(
                "Fetch key financial ratios and statements for a stock ticker. "
                "Input: just the ticker symbol, e.g. 'TCS' or 'INFY'. "
                "Returns P/E, P/B, ROE, margins, debt/equity, and quarterly revenue."
            ),
        ),
        Tool(
            name="calc_portfolio_risk",
            func=_risk_tool,
            description=(
                "Calculate portfolio risk metrics: volatility, beta, Sharpe ratio "
                "and HHI concentration index. "
                "Input JSON: {\"holdings\": [\"TCS\", \"HDFC\"], \"weights\": [0.5, 0.5]} "
                "OR CSV: 'TCS:0.4,HDFCBANK:0.3,RELIANCE:0.3'."
            ),
        ),
    ]


def get_mcp_tools_sync() -> List[Tool]:
    """Synchronous wrapper to load MCP tools (runs async event loop)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context; use fallback synchronous tools
            return _get_fallback_mcp_tools()
        return loop.run_until_complete(_load_mcp_tools())
    except Exception:
        return _get_fallback_mcp_tools()


# ── Convenience bundlers ───────────────────────────────────────────────────────

def get_fundamental_agent_tools() -> List[BaseTool]:
    """Tools for Fundamental Agent: MCP financials + price + RAG knowledge."""
    mcp_tools = get_mcp_tools_sync()
    # Filter to price and financials tools
    relevant = [t for t in mcp_tools if t.name in ("get_price", "get_financials")]
    return relevant + [get_rag_knowledge_tool()]


def get_sentiment_agent_tools() -> List[BaseTool]:
    """Tools for Sentiment Agent: Tavily news + RAG."""
    return [get_tavily_tool(), get_rag_knowledge_tool()]


def get_risk_agent_tools() -> List[BaseTool]:
    """Tools for Risk Agent: portfolio risk calculator + RAG."""
    mcp_tools = get_mcp_tools_sync()
    risk_tools = [t for t in mcp_tools if t.name == "calc_portfolio_risk"]
    return risk_tools + [get_rag_knowledge_tool()]


def get_all_tools() -> List[BaseTool]:
    """All tools bundled – useful for the Report Agent which may need context."""
    return get_mcp_tools_sync() + [get_rag_knowledge_tool(), get_tavily_tool()]
