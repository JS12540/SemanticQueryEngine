"""
engine/logger.py
================
Lightweight in-process pipeline logger.

Classes
-------
LogLevel
    Enum-like constants for log severity: DEBUG, INFO, WARNING, ERROR.

LogEntry
    Immutable dataclass for a single log event with timestamp, level,
    step name, and message.

PipelineLogger
    Collects ``LogEntry`` objects during a pipeline run and exposes them
    for display in Streamlit.  Thread-safe for a single session.

Usage
-----
    logger = PipelineLogger()
    logger.info("cache", "Checking query cache…")
    logger.info("llm",   "Calling OpenAI gpt-4o…")
    logger.warning("sql", "Validation failed – retrying.")
    logger.error("exec", "SQLite error: …")

    entries = logger.entries          # list[LogEntry]
    logger.clear()                    # reset between queries
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar


# ---------------------------------------------------------------------------
# LogLevel
# ---------------------------------------------------------------------------

class LogLevel:
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"

    # Maps level → Streamlit-friendly emoji prefix
    EMOJI: ClassVar[dict[str, str]] = {
        DEBUG:   "🔍",
        INFO:    "ℹ️",
        WARNING: "⚠️",
        ERROR:   "❌",
    }

    # Maps level → colour hex for UI rendering
    COLOUR: ClassVar[dict[str, str]] = {
        DEBUG:   "#9E9E9E",
        INFO:    "#1565C0",
        WARNING: "#E65100",
        ERROR:   "#B71C1C",
    }


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogEntry:
    """
    A single pipeline log event.

    Attributes
    ----------
    timestamp  : wall-clock time the entry was created (epoch seconds)
    level      : one of LogLevel.DEBUG/INFO/WARNING/ERROR
    step       : pipeline step name (e.g. "cache", "llm", "validate", "execute")
    message    : human-readable description of what happened
    elapsed_ms : optional elapsed time relevant to this step
    """

    timestamp:  float
    level:      str
    step:       str
    message:    str
    elapsed_ms: float | None = None

    @property
    def time_str(self) -> str:
        """HH:MM:SS.mmm string for display."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

    @property
    def emoji(self) -> str:
        return LogLevel.EMOJI.get(self.level, "•")

    @property
    def elapsed_str(self) -> str:
        if self.elapsed_ms is None:
            return ""
        return f"  ({self.elapsed_ms:.0f} ms)"

    def to_line(self) -> str:
        """Single-line text representation for plain-text logs."""
        return f"[{self.time_str}] {self.emoji} [{self.step.upper():10s}] {self.message}{self.elapsed_str}"


# ---------------------------------------------------------------------------
# PipelineLogger
# ---------------------------------------------------------------------------

class PipelineLogger:
    """
    Collects ``LogEntry`` objects produced during one NLP-to-SQL pipeline run.

    The Streamlit app creates one instance per session, passes it to the
    engine, and renders the entries after each query.

    Parameters
    ----------
    min_level : minimum severity to record (default: INFO)
                Set to LogLevel.DEBUG to capture all detail.
    """

    _LEVEL_ORDER: dict[str, int] = {
        LogLevel.DEBUG:   0,
        LogLevel.INFO:    1,
        LogLevel.WARNING: 2,
        LogLevel.ERROR:   3,
    }

    def __init__(self, min_level: str = LogLevel.INFO) -> None:
        self._min_level  = min_level
        self._entries:   list[LogEntry] = []
        self._step_start: dict[str, float] = {}

    # ── Core emit ──────────────────────────────────────────────────────

    def _emit(
        self,
        level:      str,
        step:       str,
        message:    str,
        elapsed_ms: float | None = None,
    ) -> None:
        if self._LEVEL_ORDER.get(level, 0) >= self._LEVEL_ORDER.get(self._min_level, 1):
            self._entries.append(
                LogEntry(
                    timestamp=time.time(),
                    level=level,
                    step=step,
                    message=message,
                    elapsed_ms=elapsed_ms,
                )
            )

    def debug(self, step: str, message: str, elapsed_ms: float | None = None) -> None:
        self._emit(LogLevel.DEBUG, step, message, elapsed_ms)

    def info(self, step: str, message: str, elapsed_ms: float | None = None) -> None:
        self._emit(LogLevel.INFO, step, message, elapsed_ms)

    def warning(self, step: str, message: str, elapsed_ms: float | None = None) -> None:
        self._emit(LogLevel.WARNING, step, message, elapsed_ms)

    def error(self, step: str, message: str, elapsed_ms: float | None = None) -> None:
        self._emit(LogLevel.ERROR, step, message, elapsed_ms)

    # ── Step timer helpers ─────────────────────────────────────────────

    def start_step(self, step: str) -> None:
        """Record the start time of a named pipeline step."""
        self._step_start[step] = time.perf_counter()

    def end_step(self, step: str, message: str, level: str = LogLevel.INFO) -> None:
        """Emit a log entry with elapsed time since ``start_step(step)``."""
        t0 = self._step_start.pop(step, None)
        elapsed = (time.perf_counter() - t0) * 1000 if t0 is not None else None
        self._emit(level, step, message, elapsed)

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    def entries_for_step(self, step: str) -> list[LogEntry]:
        return [e for e in self._entries if e.step == step]

    def has_errors(self) -> bool:
        return any(e.level == LogLevel.ERROR for e in self._entries)

    def has_warnings(self) -> bool:
        return any(e.level == LogLevel.WARNING for e in self._entries)

    def summary(self) -> str:
        """One-line summary: step counts and any error."""
        steps   = list(dict.fromkeys(e.step for e in self._entries))
        n_warn  = sum(1 for e in self._entries if e.level == LogLevel.WARNING)
        n_err   = sum(1 for e in self._entries if e.level == LogLevel.ERROR)
        parts   = [f"steps: {len(steps)}"]
        if n_warn:
            parts.append(f"{n_warn} warning(s)")
        if n_err:
            parts.append(f"{n_err} error(s)")
        return " | ".join(parts)

    def to_text(self) -> str:
        """Full log as a newline-joined plain-text string."""
        return "\n".join(e.to_line() for e in self._entries)

    def clear(self) -> None:
        """Reset log entries (call before each new query)."""
        self._entries.clear()
        self._step_start.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<PipelineLogger entries={len(self._entries)}>"
