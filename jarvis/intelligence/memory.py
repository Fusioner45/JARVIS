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
            log.info("Base de données FTS5 SQLite initialisée.")
        except Exception as e:
            log.error(f"Erreur initialisation Memory : {e}")

    def save_memory(self, fact_type: str, content: str, ttl_days: int = None):
        expires_at = time.time() + (ttl_days * 86400) if ttl_days else None
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO memory_fts (fact_type, content, timestamp, expires_at) VALUES (?, ?, ?, ?)",
                             (fact_type, content, time.time(), expires_at))
                conn.commit()
            log.info(f"🧠 Mémoire indexée : [{fact_type}]")
        except Exception as e:
            log.error(f"Erreur sauvegarde Memory : {e}")

    def query_memory(self, search_term: str) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT content FROM memory_fts WHERE content MATCH ?", (search_term,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            log.error(f"Erreur requête Memory : {e}")
            return []

    def get_all_context(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT fact_type, content FROM memory_fts ORDER BY timestamp DESC LIMIT 15")
                facts = [f"- {ft}: {c}" for ft, c in cursor.fetchall()]
                return "\n".join(facts) if facts else "Mémoire vide."
        except:
            return "Erreur contexte."

    def get_user_name(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT content FROM memory_fts WHERE fact_type = 'nom_utilisateur' LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else "Fusion"
        except:
            return "Fusion"

    def index_pdf(self, pdf_path: str):
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = "\n".join([p.extract_text() for p in reader.pages])
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                with sqlite3.connect(self.db_path) as conn:
                    fn = os.path.basename(pdf_path)
                    for chunk in chunks:
                        conn.execute("INSERT INTO memory_fts (fact_type, content, timestamp) VALUES (?, ?, ?)",
                                     (f"cours:{fn}", chunk, time.time()))
                    conn.commit()
            log.info(f"📚 PDF indexé : {pdf_path}")
        except Exception as e:
            log.error(f"Erreur indexation PDF : {e}")
