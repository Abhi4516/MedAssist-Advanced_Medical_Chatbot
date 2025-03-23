
import sqlite3
from datetime import datetime
from custom_logging.logger import get_logger

logger = get_logger("database_log")
DB_PATH = "chat_history.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp TEXT,
                        instruction TEXT,
                        user_input TEXT,
                        response TEXT
                    )''')
        conn.commit()
        conn.close()
        logger.info("Chat history DB initialized.")
    except Exception as e:
        logger.error(f"Error initializing chat DB: {e}")
        raise e

def add_chat_history(session_id, instruction, user_input, response):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute('''
            INSERT INTO chat_history (session_id, timestamp, instruction, user_input, response)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, timestamp, instruction, user_input, response))
        conn.commit()
        conn.close()
        logger.info(f"Chat entry added for session {session_id}.")
    except Exception as e:
        logger.error(f"Error adding chat entry: {e}")
        raise e

def get_chat_history(session_id, limit=5):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, instruction, user_input, response
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()
        if len(rows) > limit:
            rows = rows[-limit:]
        return rows
    except Exception as e:
        logger.error(f"Error retrieving chat history for session {session_id}: {e}")
        raise e

def clear_chat_history(session_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        logger.info(f"Chat history cleared for session {session_id}.")
    except Exception as e:
        logger.error(f"Error clearing chat history for {session_id}: {e}")
        raise e
