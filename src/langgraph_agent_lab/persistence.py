from __future__ import annotations

import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    kind: str = "memory",
    database_url: str | None = None,
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer.

    Supported kinds:
    - 'none': None (stateless)
    - 'memory': MemorySaver (ephemeral in-process)
    - 'sqlite': SqliteSaver (durable SQLite file or :memory:)
    - 'postgres': PostgresSaver (optional extension)
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = database_url or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is an optional extension."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
