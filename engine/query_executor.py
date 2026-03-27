"""
engine/query_executor.py
========================
SQLite wrapper with validation, execution, and result formatting.

Classes
-------
QueryResult
    Immutable dataclass holding the output of a single SQL execution –
    columns, rows, row-count, timing, and any error message.

QueryExecutor
    Manages the SQLite connection.  Exposes ``execute()``, ``validate_sql()``,
    ``get_schema_ddl()``, and schema introspection helpers.

Only SELECT / WITH queries are permitted; any write statement is rejected
before it reaches the database.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """
    Encapsulates a single SQL execution result.

    Attributes
    ----------
    sql        : the SQL string that was executed
    columns    : ordered list of column names (empty on error)
    rows       : list of row dicts – {column_name: value} (empty on error)
    row_count  : number of rows returned
    elapsed_ms : wall-clock execution time in milliseconds
    error      : error message if execution failed, else None
    """

    sql:        str
    columns:    list[str]          = field(default_factory=list)
    rows:       list[dict[str, Any]] = field(default_factory=list)
    row_count:  int                = 0
    elapsed_ms: float              = 0.0
    error:      str | None         = None

    @property
    def success(self) -> bool:
        """True when no error occurred."""
        return self.error is None

    # ── Formatting ─────────────────────────────────────────────────────

    def to_display_table(self, max_rows: int = 25) -> str:
        """
        Return a plain-text table string suitable for terminal output or
        embedding in a Streamlit code block.
        """
        if not self.success:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "(no rows returned)"

        display = self.rows[:max_rows]
        widths  = {c: len(str(c)) for c in self.columns}
        for row in display:
            for col in self.columns:
                widths[col] = max(widths[col], len(str(row.get(col, ""))))

        sep    = "+" + "+".join("-" * (w + 2) for w in widths.values()) + "+"
        header = "|" + "|".join(f" {c:<{widths[c]}} " for c in self.columns) + "|"

        lines = [sep, header, sep]
        for row in display:
            lines.append(
                "|" + "|".join(
                    f" {str(row.get(c, '')):<{widths[c]}} " for c in self.columns
                ) + "|"
            )
        lines.append(sep)

        if len(self.rows) > max_rows:
            lines.append(f"  … showing {max_rows} of {self.row_count} rows")
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Return a CSV string of all rows (for download in Streamlit)."""
        if not self.rows:
            return ",".join(self.columns)
        lines = [",".join(self.columns)]
        for row in self.rows:
            lines.append(",".join(str(row.get(c, "")) for c in self.columns))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QueryExecutor
# ---------------------------------------------------------------------------

class QueryExecutor:
    """
    Manages a read-only SQLite connection and executes validated queries.

    Parameters
    ----------
    db_path : path to the SQLite database file.
              Defaults to ``<project-root>/cashflo_sample.db``.
    """

    _ALLOWED_FIRST_WORDS: frozenset[str] = frozenset({"SELECT", "WITH", "EXPLAIN"})

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).parent.parent / "cashflo_sample.db"
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"SQLite database not found at {self._db_path}.\n"
                "Create it first: python -c \"import sqlite3; conn=sqlite3.connect('cashflo_sample.db'); "
                "conn.executescript(open('cashflo_sample_schema_and_data.sql').read()); conn.commit()\""
            )
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    # ── Validation ─────────────────────────────────────────────────────

    def validate_sql(self, sql: str) -> tuple[bool, str | None]:
        """
        Perform a dry-run syntax check using SQLite EXPLAIN.

        Returns
        -------
        (True, None)           on success
        (False, error_message) on failure
        """
        try:
            self._conn.execute(f"EXPLAIN {sql}")
            return True, None
        except sqlite3.Error as exc:
            return False, str(exc)

    def _is_read_only(self, sql: str) -> bool:
        first = sql.strip().lstrip(";").split()[0].upper() if sql.strip() else ""
        return first in self._ALLOWED_FIRST_WORDS

    # ── Execution ──────────────────────────────────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> QueryResult:
        """
        Execute *sql* and return a ``QueryResult``.

        Steps
        -----
        1. Reject non-SELECT statements (safety guard).
        2. Validate syntax via EXPLAIN.
        3. Execute and fetch all rows.
        4. Return structured result.
        """
        if not self._is_read_only(sql):
            return QueryResult(
                sql=sql,
                error="Only SELECT / WITH queries are permitted.",
            )

        ok, err = self.validate_sql(sql)
        if not ok:
            return QueryResult(sql=sql, error=f"SQL syntax error: {err}")

        t0 = time.perf_counter()
        try:
            cursor  = self._conn.execute(sql, params)
            raw     = cursor.fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            cols    = [d[0] for d in (cursor.description or [])]
            rows    = [dict(zip(cols, r)) for r in raw]
            return QueryResult(
                sql=sql,
                columns=cols,
                rows=rows,
                row_count=len(rows),
                elapsed_ms=round(elapsed, 2),
            )
        except sqlite3.Error as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return QueryResult(sql=sql, elapsed_ms=round(elapsed, 2), error=str(exc))

    # ── Schema helpers ─────────────────────────────────────────────────

    def get_schema_ddl(self) -> str:
        """Return CREATE TABLE statements for all tables (for LLM context)."""
        rows = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return "\n\n".join(r[0] for r in rows if r[0])

    def table_row_counts(self) -> dict[str, int]:
        """Return {table_name: row_count} for all tables."""
        tables = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {
            t[0]: self._conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
            for t in tables
        }

    def sample_rows(self, table: str, n: int = 3) -> list[dict]:
        """Return up to *n* sample rows from *table* as a list of dicts."""
        return self.execute(f"SELECT * FROM {table} LIMIT {n}").rows

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "QueryExecutor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<QueryExecutor db={self._db_path.name!r}>"
