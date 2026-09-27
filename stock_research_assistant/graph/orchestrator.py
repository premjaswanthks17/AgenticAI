"""
graph/orchestrator.py
──────────────────────
LangGraph orchestration for the Stock Research & Portfolio Assistant.

Graph Architecture:
  START
    │
    ▼
  [supervisor]          ← classifies query, extracts tickers
    │
    ▼ (conditional routing based on query_type)
    ├──► [rag_agent]    (conceptual questions)
    │         │
    ├──► [fundamental_agent] ─┐
    │                          ├──► [report_agent] ──► END
    ├──► [sentiment_agent]  ──┤
    │                          │
    └──► [risk_agent]       ──┘

Conditional routing rules:
  - "conceptual"       → rag_agent → report_agent
  - "fundamental"      → fundamental_agent → report_agent
  - "sentiment"        → sentiment_agent → report_agent
  - "portfolio_risk"   → risk_agent → report_agent
  - "full_research"    → fundamental_agent + sentiment_agent → report_agent
                         (parallel fan-out, then aggregate)

Three or more agents with shared state and conditional routing satisfy the
mandatory LangGraph requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.fundamental_agent import fundamental_agent_node
from agents.rag_agent import rag_agent_node
from agents.report_agent import report_agent_node
from agents.risk_agent import risk_agent_node
from agents.sentiment_agent import sentiment_agent_node
from agents.supervisor import supervisor_node
from graph.state import ResearchState

# ── Routing function ───────────────────────────────────────────────────────────

def route_after_supervisor(state: ResearchState) -> list[str]:
    """
    Conditional edge: decide which agents to call based on query_type.
    Returns a list of node names (supports parallel fan-out).
    """
    query_type = state.get("query_type", "full_research")

    routing_map = {
        "conceptual":      ["rag_agent"],
        "fundamental":     ["fundamental_agent"],
        "sentiment":       ["sentiment_agent"],
        "portfolio_risk":  ["risk_agent"],
        "full_research":   ["fundamental_agent", "sentiment_agent"],
    }

    # Default to full research if unknown
    destinations = routing_map.get(query_type, ["fundamental_agent", "sentiment_agent"])
    print(f"[Router] query_type='{query_type}' → routing to: {destinations}")
    return destinations


def route_to_report(state: ResearchState) -> str:
    """After any data-gathering agent(s) finish, always go to the report agent."""
    return "report_agent"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the full LangGraph orchestration graph.
    Returns a compiled graph ready for .invoke().
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("supervisor",        supervisor_node)
    graph.add_node("fundamental_agent", fundamental_agent_node)
    graph.add_node("sentiment_agent",   sentiment_agent_node)
    graph.add_node("risk_agent",        risk_agent_node)
    graph.add_node("rag_agent",         rag_agent_node)
    graph.add_node("report_agent",      report_agent_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Conditional routing from supervisor ────────────────────────────────────
    # Using add_conditional_edges with a list-returning function for parallel fan-out
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "fundamental_agent": "fundamental_agent",
            "sentiment_agent":   "sentiment_agent",
            "risk_agent":        "risk_agent",
            "rag_agent":         "rag_agent",
        },
    )

    # ── Converge all data-gathering agents → report_agent ─────────────────────
    graph.add_edge("fundamental_agent", "report_agent")
    graph.add_edge("sentiment_agent",   "report_agent")
    graph.add_edge("risk_agent",        "report_agent")
    graph.add_edge("rag_agent",         "report_agent")

    # ── Exit edge ─────────────────────────────────────────────────────────────
    graph.add_edge("report_agent", END)

    # Compile the graph
    compiled = graph.compile()
    return compiled


# ── Singleton compiled graph ───────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    """Return the compiled graph (singleton, built once)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── High-level run function ────────────────────────────────────────────────────

def run_research(user_query: str) -> dict:
    """
    Run the full research pipeline for a given user query.

    Args:
        user_query: Natural language question from the user.

    Returns:
        dict with keys:
          - "final_report": the formatted Markdown research note
          - "query_type":   how the query was classified
          - "tickers":      stocks identified
          - "agent_messages": execution log
    """
    graph = get_graph()

    initial_state: ResearchState = {
        "user_query": user_query,
        "tickers": [],
        "portfolio_weights": [],
        "query_type": "",
        "fundamental_data": None,
        "sentiment_data": None,
        "risk_data": None,
        "rag_answer": None,
        "final_report": None,
        "agent_messages": [],
        "error": None,
        "iteration_counts": {},
    }

    final_state = graph.invoke(initial_state)
    return {
        "final_report":    final_state.get("final_report", ""),
        "query_type":      final_state.get("query_type", ""),
        "tickers":         final_state.get("tickers", []),
        "agent_messages":  final_state.get("agent_messages", []),
        "iteration_counts": final_state.get("iteration_counts", {}),
    }
