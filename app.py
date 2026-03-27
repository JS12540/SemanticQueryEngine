"""
app.py - Cashflo NLP-to-SQL  Streamlit Application
===================================================

Single-file, all-in-one Streamlit app.  No CLI; every feature is
accessible through the UI.

OpenAI configuration (API key + model) is read exclusively from the
.env file (or environment variables) at startup.  There are no
credential inputs in the UI.

Sections
--------
Sidebar
  - Connection status (read from .env)
  - Settings (cache toggle, log level)
  - Test query runner
  - Conversation reset
  - Database info (table row counts)
  - Cache management (stats, clear)
  - Sample questions (grouped by complexity)

Main area (chat interface)
  - Conversation history (chat bubbles)
  - Per-response:
      * Assumption notice (if ambiguous)
      * KPI metrics (rows, time, tokens, cache)
      * Generated SQL (expandable)
      * Results dataframe + CSV download
      * Auto Plotly chart
      * Plain-English explanation
  - Pipeline Logs panel (per-query step-by-step trace)

Run
---
  uv run streamlit run app.py
  # or inside the venv:
  streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))

from engine.cache import QueryCache
from engine.config import Settings
from engine.logger import LogLevel, PipelineLogger
from engine.nlp_to_sql import ConversationSession, NLPQueryResult, NLPtoSQLEngine
from engine.query_executor import QueryExecutor, QueryResult


# ─────────────────────────────────────────────────────────────────────────────
# Terminal logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(filename)s:%(lineno)d  %(message)s",
    force=True,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cashflo NLP Query Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "engine":         None,   # NLPtoSQLEngine
        "session":        None,   # ConversationSession
        "messages":       [],     # list[dict] - conversation display
        "use_cache":      True,
        "log_level":      LogLevel.INFO,
        "connected":      False,
        "connect_error":  "",     # last connection error message
        "_pending_q":     None,   # question queued by sidebar sample button
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

# Test query run when "Run Test Query" is clicked
_TEST_QUERY = "How many invoices are there in total?"

SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "Simple": [
        "How many invoices were raised last month?",
        "List all vendors on the watchlist.",
        "Show me all paid invoices.",
        "What products do we buy most frequently?",
    ],
    "Joins": [
        "Show me all invoices for the Engineering department.",
        "Which vendors have overdue invoices greater than INR 1,00,000?",
        "What is the total PO value raised by each department?",
        "Which vendors supplied to the Marketing department?",
    ],
    "Aggregation": [
        "What is the total outstanding amount across all vendors?",
        "Which product has the highest total invoiced value?",
        "What was our revenue last quarter?",
        "Show me all unpaid bills.",
        "What is the average invoice value by vendor?",
    ],
    "Window Functions": [
        "Rank vendors by total invoice value.",
        "For each vendor, show the running total of payments received.",
        "Show each invoice alongside the previous invoice amount for the same vendor.",
        "Show invoice amounts with their rank within each vendor.",
    ],
    "Ambiguous / Temporal": [
        "Who are our top 5 vendors?",
        "Compare this quarter's invoice volume with last quarter.",
        "What is the approval breakdown by tier for outstanding invoices?",
        "Show me invoices from this year grouped by month.",
    ],
}


def render_sidebar() -> None:
    """Render the sidebar."""
    sb = st.sidebar

    sb.title("Cashflo NLP Query")
    sb.caption("Ask AP questions in plain English")
    sb.markdown("---")

    # -- Connection status (read-only - config lives in .env) ----------------
    sb.subheader("Connection")
    if st.session_state.connected:
        engine: NLPtoSQLEngine = st.session_state.engine
        sb.success(f"Connected  |  model: {engine._cfg.openai_model}")
        sb.caption(f"API key: {engine._cfg.openai_api_key[:8]}...")
    elif st.session_state.connect_error:
        sb.error(st.session_state.connect_error)
        sb.caption("Fix .env and restart the app.")
    else:
        sb.info("Connecting...")

    sb.markdown("---")

    # -- Settings ------------------------------------------------------------
    sb.subheader("Settings")

    use_cache = sb.toggle("Query Cache", value=st.session_state.use_cache,
                          help="Cache question->SQL mappings; fuzzy-match on reuse")
    if use_cache != st.session_state.use_cache:
        log.info("Cache setting changed to %s — rebuilding engine", use_cache)
        st.session_state.use_cache = use_cache
        _connect_engine()

    log_level_label = sb.selectbox(
        "Log detail",
        options=["INFO", "DEBUG"],
        index=0,
        help="DEBUG shows every internal step including full SQL"
    )
    st.session_state.log_level = log_level_label

    sb.markdown("---")

    # -- Test query ----------------------------------------------------------
    sb.subheader("Test Connection")
    sb.caption(f'Runs: "{_TEST_QUERY}"')
    if sb.button("Run Test Query", width="stretch",
                 disabled=not st.session_state.connected):
        st.session_state["_pending_q"] = _TEST_QUERY
        log.info("Test query queued: %s", _TEST_QUERY)
        st.rerun()

    sb.markdown("---")

    # -- Conversation control ------------------------------------------------
    sb.subheader("Conversation")
    if sb.button("Reset Conversation", width="stretch"):
        log.info("Conversation reset by user")
        st.session_state.messages = []
        if st.session_state.session:
            st.session_state.session.reset()
        st.rerun()

    sb.markdown("---")

    # -- Database info -------------------------------------------------------
    sb.subheader("Database")
    engine: NLPtoSQLEngine | None = st.session_state.engine
    if engine:
        try:
            counts = engine.executor.table_row_counts()
            for tbl, cnt in sorted(counts.items()):
                sb.caption(f"`{tbl}` - {cnt:,} rows")
        except Exception:
            sb.caption("Could not read table counts.")
    else:
        static = {
            "companies": 2, "departments": 5, "vendors": 15,
            "products": 12, "purchase_orders": 80, "po_line_items": 218,
            "grns": 68, "grn_line_items": 188, "invoices": 101,
            "invoice_line_items": 281, "payments": 38, "approval_matrix": 4,
        }
        for tbl, cnt in static.items():
            sb.caption(f"`{tbl}` - {cnt:,} rows")

    sb.markdown("---")

    # -- Cache management ----------------------------------------------------
    sb.subheader("Query Cache")
    cache_path = Path(__file__).parent / "query_cache.db"
    if cache_path.exists():
        try:
            cache = QueryCache(cache_path)
            stats = cache.stats()
            cache.close()
            c1, c2 = sb.columns(2)
            c1.metric("Stored", stats["total_cached"])
            c2.metric("Hits",   stats["total_hits"])
        except Exception:
            sb.caption("Cache not initialised yet.")
    else:
        sb.caption("No cache yet.")

    if sb.button("Clear Cache", width="stretch"):
        log.info("Cache cleared by user")
        try:
            cache = QueryCache(cache_path)
            cache.clear()
            cache.close()
            sb.success("Cache cleared.")
        except Exception as exc:
            sb.error(str(exc))

    sb.markdown("---")

    # -- Sample questions ----------------------------------------------------
    sb.subheader("Sample Questions")
    for category, questions in SAMPLE_QUESTIONS.items():
        with sb.expander(category, expanded=False):
            for q in questions:
                if st.button(
                    q,
                    key=f"sq_{hash(q)}",
                    width="stretch",
                    disabled=not st.session_state.connected,
                ):
                    st.session_state["_pending_q"] = q
                    st.rerun()


def _connect_engine() -> None:
    """Build the engine from .env settings and store in session state."""
    log.info("Connecting engine from .env")
    try:
        cfg = Settings.from_env()
        cfg.use_cache = st.session_state.use_cache
        cfg.validate()
        engine = NLPtoSQLEngine(cfg)
        st.session_state.engine        = engine
        st.session_state.session       = ConversationSession(engine)
        st.session_state.connected     = True
        st.session_state.connect_error = ""
        log.info("Engine connected  |  model=%s  db=%s  cache=%s",
                 cfg.openai_model, cfg.db_path.name, cfg.use_cache)
    except Exception as exc:
        st.session_state.engine        = None
        st.session_state.session       = None
        st.session_state.connected     = False
        st.session_state.connect_error = str(exc)
        log.error("Engine connection failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Log renderer
# ─────────────────────────────────────────────────────────────────────────────

_STEP_LABEL: dict[str, str] = {
    "pipeline": "[pipeline]",
    "ambiguity": "[ambiguity]",
    "cache":    "[cache]",
    "llm":      "[llm]",
    "validate": "[validate]",
    "fix":      "[fix]",
    "execute":  "[execute]",
    "explain":  "[explain]",
}

_LEVEL_TAG: dict[str, str] = {
    LogLevel.DEBUG:   "badge-debug",
    LogLevel.INFO:    "badge-info",
    LogLevel.WARNING: "badge-warning",
    LogLevel.ERROR:   "badge-error",
}


def render_log_panel(pipeline_log: PipelineLogger, tokens: int) -> None:
    """Render the pipeline log entries inside a Streamlit expander."""
    entries = pipeline_log.entries
    if not entries:
        return

    has_err  = pipeline_log.has_errors()
    has_warn = pipeline_log.has_warnings()

    if has_err:
        status = "[ERROR]"
    elif has_warn:
        status = "[WARN]"
    else:
        status = "[OK]"

    label = f"Pipeline Logs  {status}  ({len(entries)} events  {tokens:,} tokens)"

    with st.expander(label, expanded=has_err or has_warn):
        st.caption(pipeline_log.summary())

        st.markdown(
            """
            <style>
            .log-row       { display:flex; align-items:flex-start; margin:2px 0;
                             font-family: monospace; font-size: 0.82rem; }
            .log-time      { color:#9E9E9E; min-width:105px; }
            .log-step      { min-width:100px; font-weight:600; }
            .log-msg       { flex:1; word-break:break-all; }
            .log-elapsed   { color:#78909C; min-width:80px; text-align:right; }
            .badge-info    { color:#1565C0; }
            .badge-debug   { color:#9E9E9E; }
            .badge-warning { color:#E65100; }
            .badge-error   { color:#B71C1C; font-weight:700; }
            .log-sep       { border:none; border-top:1px solid #e0e0e0; margin:6px 0; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        rows_html = []
        prev_step = None
        for entry in entries:
            if prev_step and entry.step != prev_step:
                rows_html.append('<hr class="log-sep"/>')
            prev_step = entry.step

            step_label = _STEP_LABEL.get(entry.step, f"[{entry.step}]")
            badge      = _LEVEL_TAG.get(entry.level, "badge-info")
            elapsed    = f"{entry.elapsed_ms:.0f} ms" if entry.elapsed_ms is not None else ""

            rows_html.append(
                f'<div class="log-row">'
                f'  <span class="log-time">{entry.time_str}</span>'
                f'  <span class="log-step {badge}">{step_label}</span>'
                f'  <span class="log-msg {badge}">{entry.message}</span>'
                f'  <span class="log-elapsed">{elapsed}</span>'
                f'</div>'
            )

        st.markdown("\n".join(rows_html), unsafe_allow_html=True)

        st.download_button(
            "Download full log",
            data=pipeline_log.to_text(),
            file_name="cashflo_pipeline_log.txt",
            mime="text/plain",
            key=f"log_dl_{id(pipeline_log)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result renderer
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_chart(df: pd.DataFrame, columns: list[str]) -> str | None:
    if df.empty:
        return None
    numeric = df.select_dtypes(include="number").columns.tolist()
    text    = df.select_dtypes(exclude="number").columns.tolist()
    date_kw = any(k in " ".join(columns).lower() for k in ("date", "month", "year", "quarter"))

    if len(df) == 1 and numeric:
        return "kpi"
    if date_kw and numeric:
        return "line"
    if text and numeric and 1 < len(df) <= 20:
        return "bar"
    if text and numeric and 1 < len(df) <= 8:
        return "pie"
    if len(numeric) >= 2:
        return "scatter"
    return None


def render_result(result: NLPQueryResult, message_idx: int) -> None:
    """
    Render all widgets for a single NLPQueryResult.
    ``message_idx`` is used to make Streamlit widget keys unique.
    """
    idx = message_idx

    if result.cache_hit:
        pct = f"{result.cache_similarity * 100:.0f}%" if result.cache_similarity else "100%"
        st.info(f"Cache hit ({pct} similarity) - no LLM call made.")

    if result.assumption:
        st.warning(f"Assumption made: {result.assumption}")

    if result.retried:
        st.info("SQL was auto-corrected by the LLM after an initial error.")

    if not result.success:
        st.error(f"Error: {result.error}")
        if result.sql:
            with st.expander("Failed SQL", expanded=True):
                st.code(result.sql, language="sql")
        render_log_panel(result.log, result.tokens_used)
        return

    qr = result.query_result

    # -- KPI metrics row -----------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows returned",  qr.row_count)
    k2.metric("Query time",     f"{qr.elapsed_ms} ms")
    k3.metric("Tokens used",    f"{result.tokens_used:,}")
    k4.metric("Cache hit",      "Yes" if result.cache_hit else "No")

    # -- Query (always visible) ----------------------------------------------
    st.subheader("Generated SQL")
    st.code(result.sql, language="sql")

    # -- Results dataframe + download ----------------------------------------
    if qr.rows:
        st.subheader("Results")
        df = pd.DataFrame(qr.rows)
        st.dataframe(df, width="stretch", height=min(400, 55 + 35 * len(df)))

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buf.getvalue(),
            file_name="cashflo_result.csv",
            mime="text/csv",
            key=f"csv_{idx}",
        )

        chart_type = _suggest_chart(df, qr.columns)
        _render_chart(df, qr.columns, chart_type, idx)
    else:
        st.info("Query returned no rows.")

    # -- Explanation (always visible) ----------------------------------------
    if result.explanation:
        st.subheader("Answer")
        st.markdown(result.explanation)

    render_log_panel(result.log, result.tokens_used)


def _render_chart(
    df:         pd.DataFrame,
    columns:    list[str],
    chart_type: str | None,
    idx:        int,
) -> None:
    """Render a Plotly chart appropriate for the data shape."""
    if chart_type is None or df.empty:
        return

    numeric = df.select_dtypes(include="number").columns.tolist()
    text    = df.select_dtypes(exclude="number").columns.tolist()
    fig     = None
    title   = ""

    if chart_type == "kpi" and numeric:
        val = df[numeric[0]].iloc[0]
        st.metric(label=numeric[0], value=f"{val:,.2f}" if isinstance(val, float) else val)
        return

    elif chart_type == "line" and numeric:
        x = next((c for c in columns if any(k in c.lower() for k in ("date","month","year","quarter"))), columns[0])
        y = numeric[0]
        fig   = px.line(df, x=x, y=y, title=f"{y} over {x}", markers=True)
        title = "Line chart"

    elif chart_type == "bar" and text and numeric:
        x   = text[0]
        y   = numeric[0]
        fig = px.bar(
            df.sort_values(y, ascending=False),
            x=x, y=y,
            title=f"{y} by {x}",
            color=y,
            color_continuous_scale="Blues",
        )
        title = "Bar chart"

    elif chart_type == "pie" and text and numeric:
        fig   = px.pie(df, names=text[0], values=numeric[0],
                       title=f"Distribution of {numeric[0]}")
        title = "Pie chart"

    elif chart_type == "scatter" and len(numeric) >= 2:
        hover = text[0] if text else None
        fig   = px.scatter(df, x=numeric[0], y=numeric[1],
                           hover_name=hover,
                           title=f"{numeric[1]} vs {numeric[0]}")
        title = "Scatter plot"

    if fig:
        with st.expander(f"Auto-chart ({title})", expanded=True):
            st.plotly_chart(fig, width="stretch", key=f"chart_{idx}")


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class CashfloApp:
    """
    Top-level Streamlit application.

    Responsibilities
    ----------------
    - Initialise session state
    - Render sidebar (configuration, samples, cache)
    - Render conversation history
    - Accept new question from chat input or sidebar sample button
    - Run the NLP pipeline and render the result
    """

    def run(self) -> None:
        _init_state()
        # Auto-connect once per session on first load
        if not st.session_state.connected and not st.session_state.connect_error:
            _connect_engine()
        render_sidebar()
        self._render_main()

    # -- Main area -----------------------------------------------------------

    def _render_main(self) -> None:
        st.title("Cashflo NLP-to-SQL Query Engine")
        st.caption(
            "Ask questions about invoices, vendors, POs, payments and more "
            "in plain English.  The engine generates SQL, executes it, and "
            "explains the result."
        )

        if not st.session_state.connected:
            self._render_welcome()
            return

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(f"**{msg['content']}**")
                else:
                    render_result(msg["result"], message_idx=i)

        pending  = st.session_state.pop("_pending_q", None)
        question = st.chat_input(
            "Ask a question about your AP data...",
            disabled=not st.session_state.connected,
        ) or pending

        if question:
            self._handle_question(question)

    def _handle_question(self, question: str) -> None:
        """Process a new question: append to history, call engine, render."""
        log.info("Query received: %s", question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(f"**{question}**")

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL and executing query..."):
                session: ConversationSession = st.session_state.session
                result = session.ask(question)

            log.info(
                "Query complete  |  success=%s  rows=%s  tokens=%s  cache_hit=%s",
                result.success,
                result.query_result.row_count if result.query_result else 0,
                result.tokens_used,
                result.cache_hit,
            )
            render_result(result, message_idx=len(st.session_state.messages))

        st.session_state.messages.append({"role": "assistant", "result": result})

    # -- Welcome / not-yet-connected screen ----------------------------------

    def _render_welcome(self) -> None:
        st.markdown("---")

        if st.session_state.connect_error:
            st.error(
                f"**Could not connect to OpenAI.**\n\n"
                f"```\n{st.session_state.connect_error}\n```\n\n"
                "Check your `.env` file, set `OPENAI_API_KEY=sk-proj-...`, "
                "then restart the app."
            )
        else:
            st.info("Initialising engine from .env...")

        st.markdown("---")
        st.subheader("What you can ask")

        cols = st.columns(2)
        examples = [
            ("Simple",          "How many invoices were raised last month?"),
            ("Joins",           "Show invoices for the Engineering department."),
            ("Aggregation",     "What is the total outstanding amount?"),
            ("Synonyms",        "Show me all unpaid bills."),
            ("Window function", "Rank vendors by total invoice value."),
            ("Temporal",        "Compare this quarter with last quarter."),
            ("Business metric", "What was our revenue last quarter?"),
            ("Ambiguous",       "Who are our top 5 vendors?  (assumption stated)"),
        ]
        for i, (cat, ex) in enumerate(examples):
            cols[i % 2].markdown(f"**{cat}**\n> _{ex}_")

        st.markdown("---")
        st.subheader("Database (12 tables)")
        tbl_info = [
            ("companies",          2,   "Buyer entities"),
            ("departments",        5,   "Cost centres with annual budgets"),
            ("vendors",            15,  "Supplier master; rating, watchlist"),
            ("products",           12,  "Item catalogue with HSN codes"),
            ("purchase_orders",    80,  "POs linking company -> vendor -> dept"),
            ("po_line_items",      218, "PO line details"),
            ("grns",               68,  "Goods Receipt Notes"),
            ("grn_line_items",     188, "GRN line details"),
            ("invoices",           101, "Vendor invoices with status & deviations"),
            ("invoice_line_items", 281, "Lines with CGST/SGST/IGST breakdown"),
            ("payments",           38,  "Payments made against invoices"),
            ("approval_matrix",    4,   "Amount-based approval tiers"),
        ]
        df_schema = pd.DataFrame(tbl_info, columns=["Table", "Rows", "Description"])
        st.dataframe(df_schema, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

CashfloApp().run()
