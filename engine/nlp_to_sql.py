"""
engine/nlp_to_sql.py
====================
Core NLP-to-SQL pipeline powered by OpenAI.

Classes
-------
NLPQueryResult
    Immutable-ish dataclass holding the full output of one pipeline run –
    question, SQL, database result, explanation, cache metadata, error, and
    the pipeline log captured during the run.

NLPtoSQLEngine
    Orchestrates the full pipeline:
      1. Ambiguity detection      → state assumption
      2. Cache lookup             → skip LLM on hit
      3. LLM call (OpenAI)        → generate SQL
      4. SQL extraction + validation
      5. Self-correction retry    → feed error back to LLM
      6. Query execution
      7. Explanation generation   → second LLM call
      8. Cache store
    Emits structured log events to a ``PipelineLogger`` at every step.

ConversationSession
    Wraps ``NLPtoSQLEngine`` with multi-turn conversation memory.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .cache import QueryCache
from .config import Settings
from .logger import LogLevel, PipelineLogger
from .query_executor import QueryExecutor, QueryResult
from .semantic_layer import SemanticLayer


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior SQL analyst for an Accounts Payable (AP) automation platform.
Convert natural-language business questions into precise, correct SQLite SQL.

STRICT RULES:
1. Return ONLY the SQL inside a ```sql ... ``` code block – nothing else.
2. Use table and column names EXACTLY as they appear in the schema provided.
3. Never invent columns or tables; if the data cannot be derived, say so as a SQL comment.
4. SQLite-specific: no QUALIFY clause – use subqueries or CTEs instead. Dates are ISO strings (YYYY-MM-DD).
5. Prefer CTEs (WITH …) over deeply nested subqueries for readability.
6. Add LIMIT 100 unless the question asks for a count/aggregation or a specific number.
7. Use COALESCE to handle NULLs in aggregations.
8. Window functions RANK(), DENSE_RANK(), ROW_NUMBER(), LAG(), LEAD(), SUM() OVER () are all supported.
9. Amounts are in INR. No currency conversion needed.
10. When ambiguous, insert a comment: -- ASSUMPTION: <your assumption>
"""

_USER_PROMPT = """\
{semantic_context}

=== EXACT DATABASE SCHEMA (DDL) ===
{ddl}

=== CONVERSATION HISTORY ===
{history}

=== QUESTION ===
{question}

{ambiguity_note}

Generate the SQL query now. Return ONLY ```sql ... ``` block.
"""

_EXPLAIN_PROMPT = """\
A user asked the following business question and the database returned the result below.
Write a detailed, well-structured answer that directly addresses the question using the actual data.

Rules:
- Answer the question directly — use real names, numbers, and amounts from the result.
- Do NOT describe the SQL, joins, filters, or how the query works.
- Do NOT say "the query", "the SQL", or "the result shows" — just answer as a business analyst would.
- Cover ALL rows in the result, not just the top few. Mention specific vendors/departments/products by name.
- Include key observations: totals, highest/lowest values, patterns, outliers, rankings.
- If amounts are involved, format them clearly (e.g. INR 1,23,456).
- Use bullet points or short paragraphs to organise a longer answer.
- End with 1–2 sentences summarising the overall business implication or takeaway.

Question: {question}

Full result ({row_count} rows):
{sample}
"""

_FIX_PROMPT = """\
The SQL query below has an error. Fix it.

Original question: "{question}"

Broken SQL:
```sql
{sql}
```

SQLite error:
{error}

=== SCHEMA (DDL) ===
{ddl}

=== SEMANTIC CONTEXT (abbreviated) ===
{context}

Return ONLY the corrected SQL inside ```sql ... ```.
"""


# ---------------------------------------------------------------------------
# NLPQueryResult
# ---------------------------------------------------------------------------

@dataclass
class NLPQueryResult:
    """
    Full output of a single NLP-to-SQL pipeline run.

    Attributes
    ----------
    question         : original user question
    assumption       : assumption stated when question was ambiguous
    sql              : final SQL executed against the database
    query_result     : QueryResult from the database  (None on early failure)
    explanation      : plain-English explanation of the SQL and result
    cache_hit        : True if result came from the cache
    cache_similarity : cosine similarity score when cache_hit is True
    retried          : True if the LLM self-corrected the SQL
    error            : error message if the pipeline failed, else None
    log              : PipelineLogger populated during this run
    tokens_used      : total OpenAI tokens consumed (prompt + completion)
    """

    question:         str
    assumption:       str                   = ""
    sql:              str                   = ""
    query_result:     Optional[QueryResult] = None
    explanation:      str                   = ""
    cache_hit:        bool                  = False
    cache_similarity: Optional[float]       = None
    retried:          bool                  = False
    error:            Optional[str]         = None
    log:              PipelineLogger        = field(default_factory=PipelineLogger)
    tokens_used:      int                   = 0

    @property
    def success(self) -> bool:
        return (
            self.error is None
            and self.query_result is not None
            and self.query_result.success
        )


# ---------------------------------------------------------------------------
# NLPtoSQLEngine
# ---------------------------------------------------------------------------

class NLPtoSQLEngine:
    """
    Converts plain-English questions to executable SQLite SQL via OpenAI.

    Parameters
    ----------
    settings : ``Settings`` instance.  If not provided, ``Settings.from_env()``
               is called automatically.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._cfg      = settings or Settings.from_env()
        self._cfg.validate()
        self._client   = OpenAI(api_key=self._cfg.openai_api_key)
        self._executor = QueryExecutor(self._cfg.db_path)
        self._semantic = SemanticLayer(self._cfg.semantic_yaml)
        self._cache    = QueryCache(self._cfg.cache_db_path) if self._cfg.use_cache else None

        # Build once – doesn't change between calls
        self._semantic_ctx: str = self._semantic.build_prompt_context()
        self._ddl:          str = self._executor.get_schema_ddl()

    # ── Public API ─────────────────────────────────────────────────────

    def query(
        self,
        question:             str,
        conversation_history: list[dict] | None = None,
    ) -> NLPQueryResult:
        """
        Run the full NLP-to-SQL pipeline for *question*.

        Parameters
        ----------
        question             : natural-language question
        conversation_history : prior ``{"role": …, "content": …}`` dicts

        Returns
        -------
        ``NLPQueryResult`` with populated log
        """
        log     = PipelineLogger()
        result  = NLPQueryResult(question=question, log=log)
        history = conversation_history or []

        log.info("pipeline", f"Starting pipeline for: {question!r}")

        # ── 1. Ambiguity detection ──────────────────────────────────
        log.start_step("ambiguity")
        ambiguity_rule = self._semantic.detect_ambiguity(question)
        if ambiguity_rule:
            result.assumption = ambiguity_rule.get("default_assumption", "")
            log.end_step("ambiguity",
                         f"Ambiguous question detected → {result.assumption[:80]}",
                         LogLevel.WARNING)
        else:
            log.end_step("ambiguity", "No ambiguity detected.")

        # ── 2. Cache lookup ────────────────────────────────────────
        if self._cache:
            log.start_step("cache")
            entry = self._cache.lookup(question)
            if entry:
                sim = f"{entry.similarity * 100:.0f}%"
                log.end_step("cache", f"Cache HIT ({sim} similarity) – skipping LLM call.")
                result.sql              = entry.sql
                result.explanation      = entry.explanation
                result.cache_hit        = True
                result.cache_similarity = entry.similarity
                result.query_result     = self._executor.execute(entry.sql)
                if result.query_result.success:
                    log.info("execute",
                             f"Cached SQL executed: {result.query_result.row_count} row(s) "
                             f"in {result.query_result.elapsed_ms} ms.")
                    return result
                log.warning("cache", "Cached SQL failed – falling through to LLM.")
                result.cache_hit = False
            else:
                log.end_step("cache", "Cache MISS – will call LLM.")
        else:
            log.info("cache", "Cache disabled.")

        # ── 3. Generate SQL via LLM ───────────────────────────────
        log.start_step("llm")
        log.info("llm", f"Calling OpenAI {self._cfg.openai_model} for SQL generation…")
        sql, llm_assumption, tokens = self._generate_sql(question, history, ambiguity_rule)
        result.sql         = sql
        result.tokens_used += tokens
        if llm_assumption and not result.assumption:
            result.assumption = llm_assumption
        log.end_step(
            "llm",
            f"SQL generated ({tokens} tokens used). "
            f"{'Assumption extracted from comment.' if llm_assumption else ''}",
        )
        log.debug("llm", f"Generated SQL:\n{sql}")

        # ── 4+5. Validate → self-correction loop ──────────────────
        for attempt in range(self._cfg.max_retries + 1):
            log.start_step("validate")
            ok, err = self._executor.validate_sql(result.sql)
            if ok:
                log.end_step("validate", "SQL syntax valid (EXPLAIN passed).")
                break
            log.end_step("validate", f"SQL invalid: {err}", LogLevel.ERROR)
            if attempt < self._cfg.max_retries:
                log.info("fix", f"Asking LLM to self-correct (attempt {attempt + 1})…")
                log.start_step("fix")
                result.sql, fix_tokens = self._fix_sql(result.sql, err, question)
                result.tokens_used    += fix_tokens
                result.retried         = True
                log.end_step("fix", f"Self-corrected SQL received ({fix_tokens} tokens).")
                log.debug("fix", f"Fixed SQL:\n{result.sql}")
            else:
                result.error = (
                    f"SQL validation failed after {self._cfg.max_retries} "
                    f"self-correction attempt(s): {err}"
                )
                log.error("pipeline", result.error)
                return result

        # ── 6. Execute ────────────────────────────────────────────
        log.start_step("execute")
        result.query_result = self._executor.execute(result.sql)
        if not result.query_result.success:
            result.error = result.query_result.error
            log.end_step("execute", f"Execution failed: {result.error}", LogLevel.ERROR)
            return result
        log.end_step(
            "execute",
            f"Execution succeeded: {result.query_result.row_count} row(s) "
            f"in {result.query_result.elapsed_ms} ms.",
            LogLevel.INFO,
        )

        # ── 7. Explanation ────────────────────────────────────────
        log.start_step("explain")
        log.info("explain", "Generating plain-English explanation…")
        result.explanation, exp_tokens = self._generate_explanation(
            question, result.sql, result.query_result
        )
        result.tokens_used += exp_tokens
        log.end_step("explain", f"Explanation ready ({exp_tokens} tokens).")

        # ── 8. Cache store ────────────────────────────────────────
        if self._cache:
            self._cache.store(question, result.sql, result.explanation)
            log.info("cache", "Result stored in cache.")

        log.info("pipeline",
                 f"Pipeline complete. Total tokens: {result.tokens_used}. "
                 f"{'(self-corrected)' if result.retried else ''}")
        return result

    # ── LLM helpers ────────────────────────────────────────────────────

    def _chat(
        self,
        system:     str,
        user:       str,
        max_tokens: int = 2048,
    ) -> tuple[str, int]:
        """
        Single OpenAI chat completion.

        Returns
        -------
        (assistant_text, total_tokens_used)
        """
        messages = [{"role": "user", "content": user}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        resp = self._client.chat.completions.create(
            model=self._cfg.openai_model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=0,
        )
        text   = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens

    def _generate_sql(
        self,
        question:       str,
        history:        list[dict],
        ambiguity_rule: dict | None,
    ) -> tuple[str, str, int]:
        """Returns (sql, assumption_comment, tokens_used)."""
        ambiguity_note = ""
        if ambiguity_rule:
            assumption = ambiguity_rule.get("default_assumption", "")
            ambiguity_note = (
                f"NOTE – AMBIGUOUS QUESTION: {assumption}\n"
                "Include this as a SQL comment: -- ASSUMPTION: <assumption>"
            )

        user_msg = _USER_PROMPT.format(
            semantic_context=self._semantic_ctx,
            ddl=self._ddl,
            history=self._format_history(history),
            question=question,
            ambiguity_note=ambiguity_note,
        )
        raw, tokens = self._chat(_SYSTEM_PROMPT, user_msg, max_tokens=2048)
        sql        = self._extract_sql(raw)
        assumption = self._extract_assumption_comment(sql)
        return sql, assumption, tokens

    def _fix_sql(self, bad_sql: str, error: str, question: str) -> tuple[str, int]:
        """Returns (corrected_sql, tokens_used)."""
        user_msg = _FIX_PROMPT.format(
            question=question,
            sql=bad_sql,
            error=error,
            ddl=self._ddl,
            context=self._semantic_ctx[:3000],   # abbreviated to save tokens
        )
        raw, tokens = self._chat(_SYSTEM_PROMPT, user_msg, max_tokens=1024)
        return self._extract_sql(raw), tokens

    def _generate_explanation(self, question: str, sql: str, qr: QueryResult) -> tuple[str, int]:
        """Returns (explanation_text, tokens_used)."""
        sample = json.dumps(qr.rows, indent=2, default=str) if qr.rows else "(no rows)"
        prompt = _EXPLAIN_PROMPT.format(
            question=question,
            row_count=qr.row_count,
            sample=sample,
        )
        text, tokens = self._chat("", prompt, max_tokens=1024)
        return text.strip(), tokens

    # ── Parsing helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_sql(text: str) -> str:
        m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return re.sub(r"```[a-z]*", "", text).strip()

    @staticmethod
    def _extract_assumption_comment(sql: str) -> str:
        m = re.search(r"--\s*ASSUMPTION:\s*(.+)", sql, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "(no prior conversation)"
        return "\n".join(
            f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}"
            for t in history[-6:]
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def cache(self) -> QueryCache | None:
        return self._cache

    @property
    def executor(self) -> QueryExecutor:
        return self._executor

    @property
    def model(self) -> str:
        return self._cfg.openai_model

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        self._executor.close()
        if self._cache:
            self._cache.close()

    def __repr__(self) -> str:
        return f"<NLPtoSQLEngine model={self._cfg.openai_model!r} cache={self._cfg.use_cache}>"


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------

class ConversationSession:
    """
    Wraps ``NLPtoSQLEngine`` with multi-turn conversation memory.

    Maintains an internal history list so follow-up questions work:
        "What is our total outstanding amount?"
        "Now break that down by vendor."

    Parameters
    ----------
    engine : shared ``NLPtoSQLEngine`` instance
    """

    def __init__(self, engine: NLPtoSQLEngine) -> None:
        self._engine:  NLPtoSQLEngine = engine
        self._history: list[dict]     = []

    def ask(self, question: str) -> NLPQueryResult:
        """Ask with current conversation context; history is auto-updated."""
        result = self._engine.query(question, self._history)

        self._history.append({"role": "user", "content": question})
        if result.success and result.query_result:
            summary = (
                f"[SQL: {result.sql[:160]}] "
                f"→ {result.query_result.row_count} row(s)."
            )
        else:
            summary = f"[Query failed: {result.error}]"
        self._history.append({"role": "assistant", "content": summary})

        return result

    def reset(self) -> None:
        self._history = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def __repr__(self) -> str:
        return f"<ConversationSession turns={len(self._history) // 2}>"
