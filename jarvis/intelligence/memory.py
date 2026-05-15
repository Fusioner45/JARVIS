import sqlite3
import time
import os
from typing import List
import pypdf
from jarvis.utils.config import DB_PATH
from jarvis.utils.logger import memory_log as log

class JarvisMemory:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Activation FTS5
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(fact_type, content, timestamp, expires_at, tokenize='porter')")
                conn.commit()
            log.info("Base de données SQLite FTS5 initialisée.")
        except Exception as e:
            log.error(f"Erreur initialisation DB : {e}")

    def cleanup_obsolete(self):
        """Supprime les faits expirés."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = time.time()
                conn.execute("DELETE FROM memory_fts WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
                conn.commit()
        except Exception as e:
            log.error(f"Erreur cleanup DB : {e}")

    def save_memory(self, fact_type: str, content: str, ttl_days: int = None):
        expires_at = time.time() + (ttl_days * 86400) if ttl_days else None
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO memory_fts (fact_type, content, timestamp, expires_at) VALUES (?, ?, ?, ?)",
                             (fact_type, content, time.time(), expires_at))
                conn.commit()
                log.info(f"🧠 Mémoire sauvegardée : [{fact_type}] {content}")
        except Exception as e:
            log.error(f"Erreur sauvegarde mémoire : {e}")

    def save_task(self, task: str, due_time: str = None):
        self.save_memory("tâche", f"{task} (Échéance: {due_time})" if due_time else task, ttl_days=7)

    def index_pdf(self, pdf_path: str):
        """Extrait le texte d'un PDF et le stocke."""
        if not os.path.exists(pdf_path):
            log.error(f"Fichier PDF introuvable : {pdf_path}")
            return

        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                with sqlite3.connect(self.db_path) as conn:
                    file_name = os.path.basename(pdf_path)
                    for i, chunk in enumerate(chunks):
                        conn.execute("INSERT INTO memory_fts (fact_type, content, timestamp) VALUES (?, ?, ?)",
                                     (f"cours:{file_name}", chunk, time.time()))
                    conn.commit()
            log.info(f"📚 PDF indexé avec succès : {pdf_path}")
        except Exception as e:
            log.error(f"Erreur indexation PDF {pdf_path}: {e}")

    def query_memory(self, search_term: str) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT content FROM memory_fts WHERE content MATCH ? OR fact_type MATCH ?",
                                     (search_term, search_term))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            log.error(f"Erreur requête mémoire : {e}")
            return []

    def delete_memory(self, search_term: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memory_fts WHERE content MATCH ? OR fact_type MATCH ?",
                             (search_term, search_term))
                conn.commit()
                log.info(f"🗑️ Mémoire supprimée pour : {search_term}")
        except Exception as e:
            log.error(f"Erreur suppression mémoire : {e}")

    def get_all_context(self) -> str:
        """Récupère un résumé du contexte pour le prompt LLM."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self.cleanup_obsolete()
                cursor = conn.execute("SELECT fact_type, content FROM memory_fts ORDER BY timestamp DESC LIMIT 15")
                facts = [f"- {ft}: {c}" for ft, c in cursor.fetchall()]
                return "\n".join(facts) if facts else "Mémoire vide."
        except Exception as e:
            log.error(f"Erreur récupération contexte : {e}")
            return "Erreur accès mémoire."

    def get_user_name(self) -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT content FROM memory_fts WHERE fact_type = 'nom_utilisateur' OR content LIKE '%Je m'appelle%' LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else "Fusion"
        except:
            return "Fusion"
