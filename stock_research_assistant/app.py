"""
app.py – Stock Research & Portfolio Assistant (Streamlit)
Run:  streamlit run app.py
"""

import sys, os, importlib, datetime, traceback
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Stock Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F0F4FF !important;
    font-family: 'Inter', sans-serif !important;
    color: #1E2A3B !important;
}
.block-container {
    max-width: 900px !important;
    padding: 1.5rem 2rem 4rem !important;
}

/* ── Hide chrome ────────────────────────────────── */
#MainMenu, footer, [data-testid="stHeader"],
[data-testid="stToolbar"], .stDeployButton { display: none !important; }

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E0E7FF !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    font-size: .8rem !important; font-weight: 700 !important; color: #374151 !important;
}
[data-testid="stSidebar"] input {
    border-radius: 9px !important;
    border: 1.5px solid #C7D2FE !important;
    background: #FAFBFF !important;
    font-size: .88rem !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
}

/* ── Hero ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(130deg, #1A3C8F 0%, #2563EB 55%, #60A5FA 100%);
    border-radius: 18px; padding: 2rem 2.5rem; margin-bottom: 1.8rem;
    color: #fff; display: flex; align-items: center; gap: 1.4rem;
    box-shadow: 0 8px 36px rgba(37,99,235,.28);
}
.hero-icon  { font-size: 3rem; flex-shrink:0; }
.hero-title { font-size: 1.8rem; font-weight: 800; margin: 0 0 .2rem; letter-spacing:-.4px; }
.hero-sub   { font-size: .88rem; opacity:.85; margin:0; }
.hero-badge {
    margin-left:auto; flex-shrink:0;
    background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
    border-radius:999px; padding:.3rem .9rem;
    font-size:.73rem; font-weight:700; backdrop-filter:blur(6px);
}

/* ── Query card ───────────────────────────────────────── */
.query-card {
    background: #FFFFFF; border: 1px solid #E0E7FF;
    border-radius: 16px; padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(79,70,229,.07); margin-bottom: 1rem;
}
.sec-lbl {
    font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:#6366F1; margin:0 0 .5rem;
}

/* ── Textarea ─────────────────────────────────────────── */
[data-testid="stWidgetLabel"] { display:none !important; }
.stTextArea textarea {
    border-radius: 12px !important; border: 2px solid #C7D2FE !important;
    background: #FAFBFF !important; font-size:.95rem !important;
    padding: .8rem 1rem !important; line-height:1.65 !important;
    color: #1E2A3B !important; resize: vertical !important;
    transition: border .2s, box-shadow .2s;
}
.stTextArea textarea:focus {
    border-color:#4F46E5 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,.12) !important;
}

/* ── Buttons ──────────────────────────────────────────── */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg,#4338CA,#6366F1) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-size:.95rem !important; font-weight:700 !important;
    padding:.65rem 2rem !important;
    box-shadow:0 4px 16px rgba(99,102,241,.38) !important;
    transition: all .2s !important; width:auto !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background:linear-gradient(135deg,#3730A3,#4338CA) !important;
    box-shadow:0 6px 24px rgba(99,102,241,.48) !important;
    transform:translateY(-1px) !important;
}
[data-testid="stDownloadButton"] button {
    background:#FFFFFF !important; color:#4338CA !important;
    border:2px solid #C7D2FE !important; border-radius:10px !important;
    font-size:.88rem !important; font-weight:600 !important;
    padding:.5rem 1.2rem !important; transition:all .15s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background:#EEF2FF !important; border-color:#6366F1 !important;
}

/* ── Metrics ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background:#FFFFFF !important; border:1px solid #E0E7FF !important;
    border-radius:12px !important; padding:.9rem 1.1rem !important;
    box-shadow:0 2px 8px rgba(79,70,229,.06);
}
[data-testid="stMetricLabel"] {
    font-size:.7rem !important; font-weight:700 !important;
    color:#6366F1 !important; text-transform:uppercase; letter-spacing:.06em;
}
[data-testid="stMetricValue"] {
    font-size:1.3rem !important; font-weight:800 !important; color:#1E2A3B !important;
}

/* ── Report card ──────────────────────────────────────── */
.report-card {
    background:#FFFFFF; border:1px solid #E0E7FF; border-radius:16px;
    padding:2rem 2.5rem; box-shadow:0 2px 16px rgba(79,70,229,.08);
    font-size:.94rem; line-height:1.8; color:#1E2A3B; margin-top:.5rem;
}

/* ── Expander ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:#FFFFFF !important; border:1px solid #E0E7FF !important;
    border-radius:10px !important;
}

/* ── Pills ────────────────────────────────────────────── */
.pill-ok   { background:#D1FAE5;color:#065F46;border-radius:999px;padding:.22rem .8rem;font-size:.78rem;font-weight:700;display:inline-block; }
.pill-miss { background:#FEF3C7;color:#92400E;border-radius:999px;padding:.22rem .8rem;font-size:.78rem;font-weight:700;display:inline-block; }

/* ── Disclaimer ───────────────────────────────────────── */
.disclaimer {
    margin-top:1.5rem; padding:.85rem 1.1rem;
    background:#FFFBEB; border:1px solid #FDE68A;
    border-radius:10px; font-size:.79rem; color:#92400E; line-height:1.5;
}

hr { border-color:#E0E7FF !important; }
[data-testid="stAlert"] { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State Bootstrap ────────────────────────────────────────────────────
for key, default in [
    ("result",     None),
    ("run_error",  None),
    ("last_query", ""),
    ("needs_run",  False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ────────────────────────────────────────────────────────────────────
def run_query(query: str, gemini_key: str, tavily_key: str) -> dict:
    """Run the LangGraph research pipeline and return the result dict."""
    # Inject keys into environment (agents read config.GEMINI_API_KEY which reads os.environ)
    os.environ["GEMINI_API_KEY"] = gemini_key.strip()
    os.environ["TAVILY_API_KEY"] = tavily_key.strip()

    # Reload config so updated env vars flow through to config.GEMINI_API_KEY
    import config as _cfg
    importlib.reload(_cfg)

    # Import orchestrator AFTER config reload; reset singleton so graph picks up fresh config
    import graph.orchestrator as _orch
    _orch._compiled_graph = None
    from graph.orchestrator import run_research, build_graph
    _orch._compiled_graph = build_graph()

    return _orch.run_research(query.strip())


def build_report_md(res: dict, query: str) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tickers = ", ".join(res.get("tickers", [])) or "N/A"
    q_type  = res.get("query_type", "—").replace("_", " ").title()
    report  = res.get("final_report", "")
    return (
        f"# Stock Research Report\n"
        f"**Generated:** {now}  \n**Query:** {query}  \n"
        f"**Type:** {q_type}  \n**Tickers:** {tickers}\n\n---\n\n{report}\n\n---\n"
        f"*For educational purposes only. Not financial advice.*\n"
    )

# ── Sidebar – API Keys ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("#### 🔑 API Keys")

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza…  or  AQ.Ab8…",
        key="gk",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-dev-…",
        key="tk",
        help="Get a free key at https://app.tavily.com",
    )

    keys_ok = bool(gemini_key and tavily_key)
    if keys_ok:
        st.markdown('<span class="pill-ok">✅ Ready to run</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-miss">⚠️ Both keys required</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**What this tool does**

Runs a 6-agent LangGraph pipeline:
- 🧠 Supervisor  
- 📈 Fundamental Agent  
- 📰 Sentiment Agent  
- ⚠️ Risk Agent  
- 📚 RAG Agent  
- 📄 Report Agent  
""")
    st.caption("🎓 AgenticAI Capstone Project")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">📊</div>
  <div>
    <div class="hero-title">Stock Research &amp; Portfolio Assistant</div>
    <div class="hero-sub">Multi-agent AI · Google Gemini · RAG Knowledge Base · LangGraph Orchestration</div>
  </div>
  <div class="hero-badge">🎓 AgenticAI Capstone</div>
</div>
""", unsafe_allow_html=True)

# ── Query Input ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">🔍 Ask a question</div>', unsafe_allow_html=True)
query = st.text_area(
    "Query",
    placeholder=(
        "e.g. 'Give me a fundamental analysis of TCS'\n"
        "or   'My portfolio is 40% TCS, 30% HDFC, 30% Reliance — how risky is it?'"
    ),
    height=110,
    key="query_box",
    label_visibility="collapsed",
)

btn_col, tip_col = st.columns([1, 3])
with btn_col:
    run_btn = st.button(
        "🚀  Run Analysis",
        type="primary",
        disabled=not keys_ok,
        key="run_btn",
    )
with tip_col:
    if not keys_ok:
        st.markdown(
            "<span style='font-size:.82rem;color:#B45309;"
            "line-height:2.8;display:inline-block;'>← Enter API keys in the sidebar first</span>",
            unsafe_allow_html=True,
        )

# ── Trigger State Machine ──────────────────────────────────────────────────────
# Step 1: button click → set flag, clear old result, rerun cleanly
if run_btn and query.strip() and keys_ok:
    st.session_state.needs_run   = True
    st.session_state.result      = None
    st.session_state.run_error   = None
    st.session_state.last_query  = query
    st.rerun()

# Step 2: on the clean rerun with needs_run=True, do the actual work
if st.session_state.needs_run:
    st.session_state.needs_run = False        # prevent infinite loop
    q = st.session_state.last_query
    with st.spinner("🤖 Agents are researching… this takes 20–40 seconds."):
        try:
            st.session_state.result = run_query(q, gemini_key, tavily_key)
        except Exception:
            st.session_state.run_error = traceback.format_exc()
    # Rerun once more so results render cleanly (no spinner on screen)
    st.rerun()

# ── Error Display ──────────────────────────────────────────────────────────────
if st.session_state.run_error:
    with st.expander("❌ Error — click to see details", expanded=True):
        st.code(st.session_state.run_error, language="python")

# ── Results ────────────────────────────────────────────────────────────────────
res = st.session_state.result
if res is not None:
    st.markdown("---")

    # Metrics
    q_type  = res.get("query_type", "—")
    tickers = res.get("tickers", [])
    counts  = res.get("iteration_counts", {})

    m1, m2, m3 = st.columns(3)
    m1.metric("Query Type",  q_type.replace("_", " ").title())
    m2.metric("Tickers",     ", ".join(tickers) if tickers else "None")
    m3.metric("Agent Steps", str(sum(counts.values())) if counts else "0")

    st.markdown("<br>", unsafe_allow_html=True)

    # Report header + download
    rh_col, dl_col = st.columns([3, 1])
    with rh_col:
        st.markdown(
            "<h3 style='margin:0;font-size:1.05rem;font-weight:700;color:#1A3C8F;'>"
            "📄 Research Report</h3>",
            unsafe_allow_html=True,
        )
    with dl_col:
        last_q = st.session_state.get("last_query", "query")
        fname  = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
        st.download_button(
            label="⬇️ Download",
            data=build_report_md(res, last_q),
            file_name=fname,
            mime="text/markdown",
            key="dl_report",
        )

    # Execution log
    msgs = res.get("agent_messages", [])
    if msgs:
        with st.expander("🪵 Execution Log", expanded=False):
            for m in msgs:
                st.caption(m)

    # Report body
    report = res.get("final_report") or "_No report text was generated. See execution log above._"
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown(report)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">⚠️ <strong>Educational Use Only.</strong> '
    'This tool does not provide investment advice. Always consult a '
    'SEBI-registered financial advisor before making any investment decisions.</div>',
    unsafe_allow_html=True,
)
