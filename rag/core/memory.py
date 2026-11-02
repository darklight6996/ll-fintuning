# ==============================
# IMPORTS
# ==============================

import sqlite3
from datetime import datetime


# ==============================
# CHAT MEMORY CLASS
# ==============================

class ChatMemory:

    def __init__(
        self,
        session_id="default_session",
        db_path="sessions.db"
    ):

        self.session_id = session_id
        self.db_path = db_path

        self.conn = sqlite3.connect(
            self.db_path
        )

        self.cursor = self.conn.cursor()

        self.create_tables()

    # ==============================
    # CREATE DATABASE TABLE
    # ==============================

    def create_tables(self):

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
        """)

        self.conn.commit()

    # ==============================
    # ADD MESSAGE
    # ==============================

    def add_message(
        self,
        role,
        content
    ):

        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
        INSERT INTO messages
        (
            session_id,
            role,
            content,
            timestamp
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            self.session_id,
            role,
            content,
            timestamp
        ))

        self.conn.commit()

    # ==============================
    # ADD FULL TURN
    # ==============================

    def add_turn(
        self,
        user_message,
        assistant_message
    ):

        self.add_message(
            "user",
            user_message
        )

        self.add_message(
            "assistant",
            assistant_message
        )

    # ==============================
    # GET RECENT MEMORY
    # ==============================

    def get_recent(
        self,
        limit=6
    ):

        self.cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            self.session_id,
            limit
        ))

        rows = self.cursor.fetchall()

        rows.reverse()

        return rows

    # ==============================
    # GET FULL HISTORY
    # ==============================

    def get_full_history(self):

        self.cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (self.session_id,)
        )

        return self.cursor.fetchall()

    # ==============================
    # SEMANTIC & TEXT MEMORY SEARCH
    # ==============================

    def search_memory(self, query: str, top_k: int = 3):
        """
        Semantically search past conversation turns using SentenceTransformers embeddings.
        Falls back to keyword matching if the embedding service is unavailable.
        """
        self.cursor.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,)
        )
        rows = self.cursor.fetchall()
        if not rows:
            return []

        try:
            from core.services.embedding_service import get_embedding_service
            import numpy as np

            emb_svc = get_embedding_service()
            texts = [r[2] for r in rows]
            query_vec = emb_svc.embed(query, normalize=True)
            doc_vecs = emb_svc.embed_many(texts, normalize=True)

            scores = np.dot(doc_vecs, query_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(scores[idx])
                row = rows[idx]
                results.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "score": round(score, 4),
                })
            return results
        except Exception:
            return self.search_text(query, limit=top_k)

    def search_text(self, keyword: str, limit: int = 5):
        """
        Fast SQL text search over conversation history for exact token or keyword matches.
        """
        pattern = f"%{keyword.strip()}%"
        self.cursor.execute(
            """
            SELECT id, role, content, timestamp
            FROM messages
            WHERE session_id = ? AND content LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.session_id, pattern, limit)
        )
        rows = self.cursor.fetchall()
        return [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "timestamp": r[3],
                "score": 1.0,
            }
            for r in rows
        ]

    # ==============================
    # METRICS & EXPORT
    # ==============================

    def count(self) -> int:
        """Return total number of messages stored for this session."""
        self.cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (self.session_id,))
        row = self.cursor.fetchone()
        return row[0] if row else 0

    def get_summary(self):
        """Return summary statistics of the active conversation session."""
        self.cursor.execute(
            """
            SELECT role, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM messages
            WHERE session_id = ?
            GROUP BY role
            """,
            (self.session_id,)
        )
        rows = self.cursor.fetchall()
        user_count = 0
        assistant_count = 0
        timestamps = []
        for r in rows:
            if r[0] == "user":
                user_count = r[1]
            elif r[0] == "assistant":
                assistant_count = r[1]
            if r[2]:
                timestamps.append(r[2])
            if r[3]:
                timestamps.append(r[3])

        return {
            "session_id": self.session_id,
            "total_messages": user_count + assistant_count,
            "user_turns": user_count,
            "assistant_turns": assistant_count,
            "first_timestamp": min(timestamps) if timestamps else None,
            "last_timestamp": max(timestamps) if timestamps else None,
        }

    def export_json(self):
        """Export all messages for this session as a list of dicts."""
        self.cursor.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,)
        )
        rows = self.cursor.fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "timestamp": r[3]}
            for r in rows
        ]

    # ==============================
    # RESET SESSION
    # ==============================

    def reset_session(self):

        self.cursor.execute("""
        DELETE FROM messages
        WHERE session_id = ?
        """,
        (self.session_id,)
        )

        self.conn.commit()

    # ==============================
    # CLOSE DATABASE
    # ==============================

    def close(self):

        self.conn.close()