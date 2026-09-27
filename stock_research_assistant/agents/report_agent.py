"""
agents/report_agent.py
───────────────────────
Report Agent – final aggregation node.

Responsibilities:
  - Receive outputs from Fundamental, Sentiment and Risk agents (via shared state).
  - Synthesise a structured, professional research note.
  - Apply all guardrails: no buy/sell, include disclaimer, date the data.
  - Format as clean Markdown with clear sections.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from graph.state import ResearchState

# ── Guardrail ──────────────────────────────────────────────────────────────────

def _apply_guardrails(text: str) -> str:
    """
    Post-process the report to enforce output guardrails.
    Replaces forbidden words with neutral alternatives.
    """
    replacements = {
        r"\bbuy\b": "consider for further research",
        r"\bsell\b": "review",
        r"\bpurchase\b": "acquire",
        r"\bshort\b": "short-sell",
        r"price target of": "reference price of",
        r"target price": "reference price",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ── Report Agent Prompt ────────────────────────────────────────────────────────

REPORT_SYSTEM_PROMPT = """You are a senior equity research report writer.
Your job is to synthesise analysis from multiple specialist agents into a
clear, professional, educational research note for retail investors.

STRICT RULES:
1. NEVER use the words "buy", "sell", "purchase (as advice)", "short" or give price targets.
2. ALWAYS include the mandatory disclaimer at the END (it will be appended automatically).
3. ALWAYS state the date the data was collected in each section.
4. Be balanced – present both positive aspects and risks.
5. Write for a retail investor who may not be a finance expert.
6. Use clear section headers and bullet points where appropriate.

REQUIRED SECTIONS (include only sections for which data is available):
1. Business Overview
2. Key Financial Ratios (with interpretations)
3. News Sentiment & Recent Developments
4. Risk Profile
5. Key Open Questions (things the investor should research further)

If a section has no data, skip it gracefully.
"""

REPORT_USER_TEMPLATE = """
Synthesise the following research outputs into a complete research note.

**Company / Portfolio:** {tickers}
**Original User Query:** {user_query}
**Report Date:** {report_date}

---

### FUNDAMENTAL ANALYSIS OUTPUT:
{fundamental_data}

---

### SENTIMENT ANALYSIS OUTPUT:
{sentiment_data}

---

### RISK ANALYSIS OUTPUT:
{risk_data}

---

Please produce the final structured research note now.
"""


def report_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: Report Agent.
    Aggregates all sub-agent outputs into a final research note.
    """
    # If this was a conceptual query answered by RAG, just format that
    if state.get("query_type") == "conceptual" and state.get("rag_answer"):
        final_report = (
            "# Educational Answer\n\n"
            + (state.get("rag_answer") or "")
            + config.DISCLAIMER
        )
        log_msg = "[Report Agent] Formatted conceptual RAG answer."
        return {
            "final_report": final_report,
            "agent_messages": [log_msg],
        }

    # Build context strings
    tickers_str = ", ".join(state.get("tickers", [])) or "Portfolio"
    fundamental = state.get("fundamental_data") or "_No fundamental data available._"
    sentiment = state.get("sentiment_data") or "_No sentiment data available._"
    risk = state.get("risk_data") or "_No risk data available._"
    report_date = datetime.now().strftime("%d %B %Y")

    user_prompt = REPORT_USER_TEMPLATE.format(
        tickers=tickers_str,
        user_query=state["user_query"],
        report_date=report_date,
        fundamental_data=fundamental,
        sentiment_data=sentiment,
        risk_data=risk,
    )

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0.1,   # slight creativity for better prose
        google_api_key=config.GEMINI_API_KEY,
    )

    messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    if isinstance(response.content, str):
        raw_report = response.content
    elif isinstance(response.content, list):
        raw_report = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content)
    else:
        raw_report = str(response.content)

    # Apply guardrails (belt-and-suspenders check)
    report_with_guardrails = _apply_guardrails(raw_report)

    # Append mandatory disclaimer
    final_report = report_with_guardrails + config.DISCLAIMER

    log_msg = (
        f"[Report Agent] Final research note produced for {tickers_str}. "
        f"Length: {len(final_report)} characters."
    )
    print(log_msg)

    return {
        "final_report": final_report,
        "agent_messages": [log_msg],
    }
