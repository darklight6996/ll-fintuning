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