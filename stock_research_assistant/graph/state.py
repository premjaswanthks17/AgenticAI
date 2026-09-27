"""
graph/state.py
──────────────
Shared state TypedDict used across all LangGraph nodes.
Every agent reads from and writes to this single state object,
enabling clean data flow and conditional routing.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
import operator


def _merge_dicts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left) if left else {}
    if right:
        merged.update(right)
    return merged


class ResearchState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    Fields:
        user_query          : Original question from the user.
        tickers             : List of stock tickers detected (e.g. ["TCS", "INFY"]).
        portfolio_weights   : Weights for each ticker (0–1), empty for single-stock.
        query_type          : Routing label assigned by the Supervisor.
        fundamental_data    : Raw output from the Fundamental Agent.
        sentiment_data      : Raw output from the Sentiment Agent.
        risk_data           : Raw output from the Risk Agent.
        rag_answer          : Answer produced by direct RAG lookup.
        final_report        : Formatted research note from the Report Agent.
        agent_messages      : Accumulated agent log / reasoning trace.
        error               : Any error message for graceful degradation.
        iteration_counts    : Per-agent ReAct iteration tracking.
    """

    user_query: str
    tickers: list[str]
    portfolio_weights: list[float]
    query_type: str          # "fundamental" | "sentiment" | "portfolio_risk" | "conceptual" | "full_research"
    fundamental_data: Optional[str]
    sentiment_data: Optional[str]
    risk_data: Optional[str]
    rag_answer: Optional[str]
    final_report: Optional[str]
    # Annotated with operator.add so multiple agents can append without overwriting
    agent_messages: Annotated[list[str], operator.add]
    error: Optional[str]
    iteration_counts: Annotated[dict[str, int], _merge_dicts]

