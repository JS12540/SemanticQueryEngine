"""
engine
======
Cashflo NLP-to-SQL engine package.

Public re-exports
-----------------
    from engine import NLPtoSQLEngine, ConversationSession
    from engine import QueryExecutor, QueryResult
    from engine import SemanticLayer
    from engine import QueryCache
    from engine import PipelineLogger, LogEntry, LogLevel
"""

from .cache import CacheEntry, QueryCache
from .config import Settings
from .logger import LogEntry, LogLevel, PipelineLogger
from .nlp_to_sql import ConversationSession, NLPQueryResult, NLPtoSQLEngine
from .query_executor import QueryExecutor, QueryResult
from .semantic_layer import SemanticLayer, TemporalResolver

__all__ = [
    # Core pipeline
    "NLPtoSQLEngine",
    "ConversationSession",
    "NLPQueryResult",
    # DB
    "QueryExecutor",
    "QueryResult",
    # Semantic layer
    "SemanticLayer",
    "TemporalResolver",
    # Cache
    "QueryCache",
    "CacheEntry",
    # Logging
    "PipelineLogger",
    "LogEntry",
    "LogLevel",
    # Config
    "Settings",
]
