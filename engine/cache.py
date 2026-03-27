"""
engine/cache.py
===============
Persistent, similarity-aware query cache backed by SQLite.

Classes
-------
CacheEntry
    Lightweight dataclass representing a single cached record.

QueryCache
    Manages the cache database.  Supports exact-hash lookup for speed,
    falls back to TF-IDF cosine similarity for fuzzy matching of
    paraphrased questions (e.g. "total revenue" ≈ "what is our revenue?").

Cache schema (table: query_cache)
----------------------------------
    question_hash TEXT PRIMARY KEY  – SHA-256 of the normalised question
    question      TEXT              – original question text
    sql           TEXT              – generated SQL
    explanation   TEXT              – plain-English explanation
    hit_count     INTEGER           – times this entry was reused
    created_at    TIMESTAMP
    last_used_at  TIMESTAMP
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single record returned by QueryCache.lookup()."""
    question:   str
    sql:        str
    explanation: str
    similarity: float   # 1.0 = exact match, <1.0 = fuzzy


# ---------------------------------------------------------------------------
# Internal similarity helpers (module-level, no state)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


def _cosine_similarity(a: list[str], b: list[str]) -> float:
    """TF-cosine similarity between two token lists, in [0.0, 1.0]."""
    tf_a: Counter = Counter(a)
    tf_b: Counter = Counter(b)
    vocab = set(tf_a) | set(tf_b)
    dot   = sum(tf_a[w] * tf_b[w] for w in vocab)
    mag_a = math.sqrt(sum(v * v for v in tf_a.values()))
    mag_b = math.sqrt(sum(v * v for v in tf_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# QueryCache
# ---------------------------------------------------------------------------

class QueryCache:
    """
    Persistent cache for question → SQL mappings.

    Parameters
    ----------
    db_path              : path to the SQLite cache file.
                           Defaults to ``<project-root>/query_cache.db``.
    similarity_threshold : minimum cosine similarity to count as a cache hit.
                           0.85 works well for paraphrased business questions.
    """

    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        db_path: str | Path | None = None,
        similarity_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if db_path is None:
            db_path = Path(__file__).parent.parent / "query_cache.db"
        self._db_path  = Path(db_path)
        self._threshold = similarity_threshold
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ── Schema ─────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS query_cache (
                question_hash TEXT PRIMARY KEY,
                question      TEXT NOT NULL,
                sql           TEXT NOT NULL,
                explanation   TEXT DEFAULT '',
                hit_count     INTEGER DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._conn.commit()

    # ── Lookup ─────────────────────────────────────────────────────────

    def lookup(self, question: str) -> Optional[CacheEntry]:
        """
        Search the cache for a question matching *question*.

        Strategy
        --------
        1. Exact hash match (O(1)).
        2. Cosine similarity scan over all cached questions.
           Returns the best match above ``similarity_threshold``.

        Returns ``CacheEntry`` on hit, ``None`` on miss.
        """
        q_hash = _sha256(_normalise(question))

        # Exact match
        row = self._conn.execute(
            "SELECT * FROM query_cache WHERE question_hash = ?", (q_hash,)
        ).fetchone()
        if row:
            self._bump(q_hash)
            return CacheEntry(
                question=row["question"],
                sql=row["sql"],
                explanation=row["explanation"] or "",
                similarity=1.0,
            )

        # Fuzzy match
        all_rows = self._conn.execute(
            "SELECT question_hash, question, sql, explanation FROM query_cache"
        ).fetchall()

        q_tokens  = _tokenise(question)
        best_score = 0.0
        best_row   = None
        for cached in all_rows:
            score = _cosine_similarity(q_tokens, _tokenise(cached["question"]))
            if score > best_score:
                best_score = score
                best_row   = cached

        if best_row and best_score >= self._threshold:
            self._bump(best_row["question_hash"])
            return CacheEntry(
                question=best_row["question"],
                sql=best_row["sql"],
                explanation=best_row["explanation"] or "",
                similarity=round(best_score, 3),
            )

        return None

    # ── Store ──────────────────────────────────────────────────────────

    def store(self, question: str, sql: str, explanation: str = "") -> None:
        """
        Persist a question → SQL mapping.
        Upserts on conflict (same normalised question hash).
        """
        q_hash = _sha256(_normalise(question))
        self._conn.execute(
            """
            INSERT INTO query_cache (question_hash, question, sql, explanation)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(question_hash) DO UPDATE SET
                sql          = excluded.sql,
                explanation  = excluded.explanation,
                last_used_at = CURRENT_TIMESTAMP
            """,
            (q_hash, question, sql, explanation),
        )
        self._conn.commit()

    # ── Admin ──────────────────────────────────────────────────────────

    def _bump(self, q_hash: str) -> None:
        self._conn.execute(
            """
            UPDATE query_cache
               SET hit_count    = hit_count + 1,
                   last_used_at = CURRENT_TIMESTAMP
             WHERE question_hash = ?
            """,
            (q_hash,),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        """Return {total_cached, total_hits}."""
        row = self._conn.execute(
            "SELECT COUNT(*) total, COALESCE(SUM(hit_count),0) hits FROM query_cache"
        ).fetchone()
        return {"total_cached": row["total"], "total_hits": row["hits"]}

    def list_entries(self) -> list[dict]:
        """Return all cached entries ordered by hit_count desc."""
        rows = self._conn.execute(
            "SELECT question, hit_count, created_at, last_used_at "
            "FROM query_cache ORDER BY hit_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        """Delete all cached entries."""
        self._conn.execute("DELETE FROM query_cache")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"<QueryCache path={self._db_path.name!r} "
            f"entries={s['total_cached']} hits={s['total_hits']}>"
        )
