import sqlite3
from datetime import datetime
from custom_logging.logger import get_logger

logger = get_logger("session_db_log")
DB_PATH = "session_history.db"

def init_session_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                creation_time TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Session DB initialized.")
    except Exception as e:
        logger.error(f"Error initializing session DB: {e}")
        raise e

def add_session(session_id, creation_time):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO sessions (session_id, creation_time) 
            VALUES (?, ?)
        ''', (session_id, creation_time))
        conn.commit()
        conn.close()
        logger.info(f"Session {session_id} added at {creation_time}.")
    except Exception as e:
        logger.error(f"Error adding session {session_id}: {e}")
        raise e

def get_sessions(limit=6):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT session_id, creation_time 
            FROM sessions 
            ORDER BY creation_time DESC 
            LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        raise e

def clear_old_sessions(max_sessions=6):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT session_id FROM sessions ORDER BY creation_time DESC')
        rows = c.fetchall()
        all_sessions = [r[0] for r in rows]
        if len(all_sessions) > max_sessions:
            to_delete = all_sessions[max_sessions:]
            for sid in to_delete:
                c.execute('DELETE FROM sessions WHERE session_id = ?', (sid,))
            conn.commit()
        conn.close()
        logger.info("Old sessions cleared if needed.")
    except Exception as e:
        logger.error(f"Error clearing old sessions: {e}")
        raise e
