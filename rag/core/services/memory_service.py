"""
core/services/memory_service.py

Factory / registry for ChatMemory instances, keyed by session_id.

Replaces direct ChatMemory construction scattered across callers.
Each unique session_id gets exactly one ChatMemory instance per process,
preventing SQLite connection leaks.

Usage:
    from core.services.memory_service import get_memory
    mem = get_memory("my_session")
    mem.add_turn("what is SUID?", "SUID stands for ...")
    recent = mem.get_recent(limit=6)
"""

from typing import Dict, Optional

_REGISTRY: Dict[str, "ChatMemoryWrapper"] = {}


class ChatMemoryWrapper:
    """
    Thin wrapper around ChatMemory that defers import and provides
    a consistent interface regardless of whether the DB is available.
    """

    def __init__(self, session_id: str, db_path: Optional[str] = None):
        self.session_id = session_id
        self._memory = None
        self._load(session_id=session_id, db_path=db_path)

    def _load(self, session_id: str, db_path: Optional[str]) -> None:
        try:
            from core.memory import ChatMemory
            from core.config import DATABASE_PATH
            path = db_path or DATABASE_PATH
            self._memory = ChatMemory(session_id=session_id, db_path=path)
        except Exception as exc:
            print(f"[!] MemoryService: could not open DB for session '{session_id}' ({exc})")
            self._memory = None

    # ------------------------------------------------------------------
    # Delegating interface — same as ChatMemory
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._memory is not None

    def add_message(self, role: str, content: str) -> None:
        if self._memory:
            self._memory.add_message(role, content)

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        if self._memory:
            self._memory.add_turn(user_message, assistant_message)

    def get_recent(self, limit: int = 6):
        if self._memory:
            return self._memory.get_recent(limit=limit)
        return []

    def get_full_history(self):
        if self._memory:
            return self._memory.get_full_history()
        return []

    def search_memory(self, query: str, top_k: int = 3):
        if self._memory:
            return self._memory.search_memory(query=query, top_k=top_k)
        return []

    def search_text(self, keyword: str, limit: int = 5):
        if self._memory:
            return self._memory.search_text(keyword=keyword, limit=limit)
        return []

    def count(self) -> int:
        if self._memory:
            return self._memory.count()
        return 0

    def get_summary(self):
        if self._memory:
            return self._memory.get_summary()
        return {}

    def export_json(self):
        if self._memory:
            return self._memory.export_json()
        return []

    def reset_session(self) -> None:
        if self._memory:
            self._memory.reset_session()

    def close(self) -> None:
        if self._memory:
            self._memory.close()

    @property
    def raw(self):
        """Direct access to the underlying ChatMemory (for advanced use)."""
        return self._memory


def get_memory(
    session_id: str = "default_session",
    db_path: Optional[str] = None,
) -> ChatMemoryWrapper:
    """
    Return a per-session ChatMemory wrapper.

    Each session_id maps to exactly one ChatMemoryWrapper per process.
    Creating the same session_id twice returns the cached instance.

    Parameters
    ----------
    session_id : str
        Unique identifier for the chat session.
    db_path : str, optional
        Override the SQLite database path from config.
    """
    global _REGISTRY
    if session_id not in _REGISTRY:
        _REGISTRY[session_id] = ChatMemoryWrapper(session_id=session_id, db_path=db_path)
    return _REGISTRY[session_id]


def close_all() -> None:
    """Close all open memory connections (call at process exit if needed)."""
    global _REGISTRY
    for wrapper in _REGISTRY.values():
        wrapper.close()
    _REGISTRY.clear()
