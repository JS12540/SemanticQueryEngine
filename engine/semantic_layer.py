"""
engine/semantic_layer.py
========================
Loads and exposes the semantic_layer.yaml configuration.

The ``SemanticLayer`` class is the single source of truth for:
- Table and column descriptions
- Join paths (relationships)
- Business metrics  (revenue, outstanding, …)
- Global synonyms   (bills → invoices, unpaid → status filter)
- Temporal helpers  (last_quarter, this_month → ISO date ranges)
- Ambiguity rules   (top vendors, outstanding)

It also generates the **prompt context block** – a compact, structured string
injected into every LLM prompt so the model understands business domain
vocabulary without having to reverse-engineer raw DDL.
"""

from __future__ import annotations

import re
import textwrap
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


class TemporalResolver:
    """
    Resolves plain-English time phrases to concrete ISO date ranges.

    All calculations are based on today's date so the same YAML config
    works correctly over time without manual updates.
    """

    def __init__(self) -> None:
        self._today: date = date.today()

    # ── Public helpers ─────────────────────────────────────────────────

    def today(self) -> str:
        return self._today.isoformat()

    def this_month(self) -> tuple[str, str]:
        """First day of current month → today."""
        start = self._today.replace(day=1)
        return start.isoformat(), self._today.isoformat()

    def last_month(self) -> tuple[str, str]:
        """First → last day of the previous calendar month."""
        first_this = self._today.replace(day=1)
        last_prev   = first_this - timedelta(days=1)
        first_prev  = last_prev.replace(day=1)
        return first_prev.isoformat(), last_prev.isoformat()

    def this_quarter(self) -> tuple[str, str]:
        """First day of current calendar quarter → today."""
        q_start_month = ((self._today.month - 1) // 3) * 3 + 1
        start = date(self._today.year, q_start_month, 1)
        return start.isoformat(), self._today.isoformat()

    def last_quarter(self) -> tuple[str, str]:
        """First → last day of the previous calendar quarter."""
        q_start_month = ((self._today.month - 1) // 3) * 3 + 1
        if q_start_month == 1:
            prev_month = 10
            prev_year  = self._today.year - 1
        else:
            prev_month = q_start_month - 3
            prev_year  = self._today.year
        start = date(prev_year, prev_month, 1)
        end   = date(self._today.year, q_start_month, 1) - timedelta(days=1)
        return start.isoformat(), end.isoformat()

    def this_year(self) -> tuple[str, str]:
        start = date(self._today.year, 1, 1)
        return start.isoformat(), self._today.isoformat()

    def last_year(self) -> tuple[str, str]:
        start = date(self._today.year - 1, 1, 1)
        end   = date(self._today.year - 1, 12, 31)
        return start.isoformat(), end.isoformat()

    def as_context_dict(self) -> dict[str, tuple[str, str] | str]:
        """Return all resolved temporal ranges as a dict for prompt injection."""
        return {
            "today":        self.today(),
            "this_month":   self.this_month(),
            "last_month":   self.last_month(),
            "this_quarter": self.this_quarter(),
            "last_quarter": self.last_quarter(),
            "this_year":    self.this_year(),
            "last_year":    self.last_year(),
        }


class SemanticLayer:
    """
    Parses semantic_layer.yaml and provides helper methods for the
    NLP-to-SQL pipeline.

    Parameters
    ----------
    yaml_path : path to semantic_layer.yaml.
                Defaults to ``<project-root>/semantic_layer.yaml``.
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / "semantic_layer.yaml"
        self._path: Path = Path(yaml_path)
        with open(self._path, encoding="utf-8") as fh:
            self._cfg: dict[str, Any] = yaml.safe_load(fh)

        self.tables:          dict[str, Any] = self._cfg.get("tables", {})
        self.metrics:         dict[str, Any] = self._cfg.get("metrics", {})
        self.synonyms:        dict[str, str] = self._cfg.get("synonyms", {})
        self.ambiguity_rules: dict[str, Any] = self._cfg.get("ambiguity_rules", {})
        self.temporal_cfg:    dict[str, Any] = self._cfg.get("temporal", {})

        _rel = self._cfg.get("relationships", {})
        if isinstance(_rel, dict):
            self.direct_joins: list[dict] = _rel.get("direct", [])
            self.multi_hops:   list[dict] = _rel.get("multi_hop", [])
        else:
            self.direct_joins = _rel
            self.multi_hops   = []

        # Pre-compute lower-case synonym lookup
        self._syn_map: dict[str, str] = {
            k.lower(): v for k, v in self.synonyms.items()
        }

        self._temporal = TemporalResolver()

    # ── Synonym / metric helpers ───────────────────────────────────────

    def resolve_synonym(self, word: str) -> str | None:
        """Return canonical term for *word*, or None if unknown."""
        return self._syn_map.get(word.lower())

    def find_metric(self, name: str) -> dict | None:
        """Return metric definition by name or synonym, or None."""
        name_lower = name.lower()
        if name_lower in self.metrics:
            return self.metrics[name_lower]
        for m_name, m_def in self.metrics.items():
            if name_lower in [s.lower() for s in m_def.get("synonyms", [])]:
                return m_def
        return None

    # ── Temporal helpers ───────────────────────────────────────────────

    @property
    def temporal(self) -> TemporalResolver:
        """Access the temporal resolver for date range computations."""
        return self._temporal

    # ── Ambiguity detection ────────────────────────────────────────────

    def detect_ambiguity(self, question: str) -> dict | None:
        """
        Check whether *question* matches a known ambiguity trigger.

        The trigger phrase words are matched as a subset of the question
        words, so "top vendors" matches "Who are our top 5 vendors?".

        Returns the ambiguity rule dict (containing *default_assumption*)
        or None when the question is unambiguous.
        """
        q_words = set(re.findall(r"\w+", question.lower()))
        for rule in self.ambiguity_rules.values():
            trigger = rule.get("trigger", "").lower()
            if not trigger:
                continue
            if set(trigger.split()).issubset(q_words):
                return rule
        return None

    # ── Prompt context builder ─────────────────────────────────────────

    def build_prompt_context(self) -> str:
        """
        Build a compact structured string suitable for LLM prompt injection.

        Contains: tables+columns, relationships, metrics, synonyms, and
        the resolved temporal date ranges for today's date.
        """
        lines: list[str] = []

        # Tables & columns
        lines.append("=== DATABASE TABLES ===")
        for tbl_name, tbl in self.tables.items():
            lines.append(f"\nTable: {tbl_name}")
            lines.append(f"  Description: {tbl.get('description', '')}")
            syns = tbl.get("synonyms", [])
            if syns:
                lines.append(f"  Synonyms: {', '.join(syns)}")
            lines.append("  Columns:")
            for col, meta in tbl.get("columns", {}).items():
                if isinstance(meta, dict):
                    col_type = meta.get("type", "")
                    desc     = meta.get("desc", "")
                    vals     = meta.get("values", [])
                    val_str  = f" [{', '.join(vals)}]" if vals else ""
                    lines.append(f"    {col} ({col_type}){val_str}: {desc}")
                else:
                    lines.append(f"    {col}: {meta}")

        # Relationships
        lines.append("\n=== JOIN RELATIONSHIPS ===")
        for rel in self.direct_joins:
            lines.append(f"  {rel['from']} → {rel['to']}  ON  {rel['join']}")
        for hop in self.multi_hops:
            lines.append(f"  [multi-hop] {hop.get('path', '')}")
            lines.append(f"    via: {hop.get('via', '')}")

        # Business metrics
        lines.append("\n=== BUSINESS METRICS ===")
        for m_name, m_def in self.metrics.items():
            lines.append(f"  {m_name}: {m_def.get('description', '')}")
            lines.append(f"    SQL fragment: {m_def.get('sql', '')}")
            syns = m_def.get("synonyms", [])
            if syns:
                lines.append(f"    Synonyms: {', '.join(syns)}")

        # Global synonyms
        lines.append("\n=== VOCABULARY / SYNONYMS ===")
        for word, meaning in self.synonyms.items():
            lines.append(f"  '{word}' → {meaning}")

        # Temporal
        lines.append("\n=== TEMPORAL EXPRESSIONS (resolved for today) ===")
        td = self._temporal.as_context_dict()
        lines.append(f"  today:        {td['today']}")
        for key in ("this_month", "last_month", "this_quarter", "last_quarter",
                    "this_year", "last_year"):
            start, end = td[key]
            lines.append(f"  {key:<14}: {start}  →  {end}")

        return "\n".join(lines)

    # ── Introspection ──────────────────────────────────────────────────

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def __repr__(self) -> str:
        return f"<SemanticLayer path={self._path.name!r} tables={len(self.tables)}>"
