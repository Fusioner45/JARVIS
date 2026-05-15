import sqlite3
import time
import os
from typing import List
import pypdf
from jarvis.utils.config import DB_PATH
from jarvis.utils.logger import system_log as log

class JarvisMemory:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(fact_type, content, timestamp, expires_at, tokenize='porter')")
                conn.commit()
            log.info("Mémoire SQLite FTS5 initialisée.")
        except Exception as e: log.error(f"Memory Init Error: {e}")

    def save_memory(self, fact_type: str, content: str, ttl_days: int = None):
        exp = time.time() + (ttl_days * 86400) if ttl_days else None
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO memory_fts VALUES (?, ?, ?, ?)", (fact_type, content, time.time(), exp))
                conn.commit()
            log.info(f"🧠 Mémoire sauvegardée : {fact_type}")
        except Exception as e: log.error(f"Memory Save Error: {e}")

    def query_memory(self, term: str) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT content FROM memory_fts WHERE content MATCH ?", (term,))
                return [r[0] for r in cur.fetchall()]
        except: return []

    def get_all_context(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT fact_type, content FROM memory_fts ORDER BY timestamp DESC LIMIT 15")
                return "\n".join([f"- {ft}: {c}" for ft, c in cur.fetchall()])
        except: return "Mémoire vide."

    def get_user_name(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT content FROM memory_fts WHERE fact_type = 'nom_utilisateur' LIMIT 1")
                row = cur.fetchone()
                return row[0] if row else "Fusion"
        except: return "Fusion"
