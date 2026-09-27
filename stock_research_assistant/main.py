"""
main.py
───────
Entry point for the Stock Research & Portfolio Assistant.

Usage:
    # Interactive mode (REPL)
    python main.py

    # Single query mode
    python main.py --query "Give me a research summary of Infosys"

    # Rebuild the RAG vector store
    python main.py --rebuild-rag

    # Visualise the LangGraph
    python main.py --show-graph
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config

# ── Sample queries (from the problem statement) ───────────────────────────────

SAMPLE_QUERIES = [
    "Give me a research summary of Infosys.",
    "My portfolio is 40% TCS, 30% HDFC Bank and 30% Reliance. How risky is it?",
    "What does a P/E ratio of 35 mean for this company?",
    "What is the recent news sentiment around Tata Motors?",
]

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        📊  STOCK RESEARCH & PORTFOLIO ASSISTANT  📊          ║
║          AgenticAI Capstone – Finance Domain                  ║
╚══════════════════════════════════════════════════════════════╝

⚠️  This tool is for EDUCATIONAL PURPOSES ONLY.
    It does NOT provide investment advice.

Sample queries:
  1. Give me a research summary of Infosys.
  2. My portfolio is 40% TCS, 30% HDFC Bank and 30% Reliance. How risky is it?
  3. What does a P/E ratio of 35 mean for this company?
  4. What is the recent news sentiment around Tata Motors?

Type 'quit' or 'exit' to stop.
Type 'sample <N>' to run sample query N (1-4).
Type 'graph' to print the LangGraph structure.
"""


def print_report(result: dict) -> None:
    """Pretty-print the research report to the console."""
    print("\n" + "═" * 70)
    print(f"  Query Type : {result['query_type'].upper()}")
    print(f"  Tickers    : {', '.join(result['tickers']) or 'N/A'}")
    print(f"  Iterations : {result['iteration_counts']}")
    print("═" * 70 + "\n")
    print(result.get("final_report", "No report generated."))
    print("\n" + "─" * 70)


def run_single_query(query: str) -> None:
    """Run a single query and print the result."""
    from graph.orchestrator import run_research
    print(f"\n🔍 Processing: {query}\n")
    result = run_research(query)
    print_report(result)


def rebuild_rag() -> None:
    """Force-rebuild the RAG vector store."""
    from rag.knowledge_base import build_vectorstore
    print("🔄 Rebuilding RAG vector store …")
    build_vectorstore(force_rebuild=True)
    print("✅ Vector store rebuilt successfully.")


def show_graph() -> None:
    """Print the LangGraph ASCII representation."""
    try:
        from graph.orchestrator import build_graph
        g = build_graph()
        print("\nLangGraph Structure:")
        print(g.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not draw graph: {e}")
        print("Graph nodes: supervisor → [fundamental_agent | sentiment_agent | risk_agent | rag_agent] → report_agent → END")


def interactive_mode() -> None:
    """Run the assistant in interactive REPL mode."""
    from graph.orchestrator import run_research

    print(WELCOME_BANNER)

    # Ensure RAG store is built
    try:
        from rag.knowledge_base import build_vectorstore
        build_vectorstore()
    except Exception as e:
        print(f"⚠️  RAG setup warning: {e}")

    while True:
        try:
            user_input = input("\n💬 Your query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Remember: always consult a financial advisor. 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Remember: always consult a financial advisor. 👋")
            break

        if user_input.lower() == "graph":
            show_graph()
            continue

        # Handle sample query shortcuts
        if user_input.lower().startswith("sample"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1]) - 1
                if 0 <= idx < len(SAMPLE_QUERIES):
                    user_input = SAMPLE_QUERIES[idx]
                    print(f"▶ Running: {user_input}")
                else:
                    print(f"Sample {parts[1]} not found. Choose 1–{len(SAMPLE_QUERIES)}.")
                    continue

        try:
            result = run_research(user_input)
            print_report(result)
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock Research & Portfolio Assistant – AgenticAI Capstone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python main.py
              python main.py --query "Give me a research summary of Infosys"
              python main.py --rebuild-rag
              python main.py --show-graph
        """),
    )
    parser.add_argument("--query",       type=str, help="Run a single query and exit.")
    parser.add_argument("--rebuild-rag", action="store_true", help="Rebuild the RAG vector store.")
    parser.add_argument("--show-graph",  action="store_true", help="Show LangGraph structure.")
    parser.add_argument("--sample",      type=int, choices=[1, 2, 3, 4],
                        help="Run one of the 4 sample queries (1–4).")

    args = parser.parse_args()

    # Validate config (will raise if keys are missing)
    try:
        config.validate_config()
    except EnvironmentError as e:
        print(f"\n❌ Configuration Error: {e}\n")
        sys.exit(1)

    if args.rebuild_rag:
        rebuild_rag()
        return

    if args.show_graph:
        show_graph()
        return

    if args.sample:
        query = SAMPLE_QUERIES[args.sample - 1]
        run_single_query(query)
        return

    if args.query:
        run_single_query(args.query)
        return

    # Default: interactive REPL
    interactive_mode()


if __name__ == "__main__":
    main()
