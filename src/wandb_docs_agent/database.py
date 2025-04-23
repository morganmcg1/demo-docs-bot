# src/wandb_docs_agent/database.py
import sqlite3
import logging
import json
from typing import Tuple, Optional
from pathlib import Path
from wandb_docs_agent.models import SupportTicketContext

DATABASE_FILE = Path("databases/conversation_state.sqlite")
DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE.absolute(), check_same_thread=False) # check_same_thread=False for FastAPI usage
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def init_db():
    """Initializes the database and creates the conversation_state table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_state (
                    conversation_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    last_response_id TEXT,
                    context_json TEXT NOT NULL
                )
            """)
            conn.commit()
            logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise # Re-raise the exception to halt startup if DB init fails

def save_state_to_db(
    conversation_id: str,
    context: dict,
    agent_name: str,
    last_response_id: str,
):
    """
    Save the entire context as a JSON blob (no validation) for maximum flexibility.
    """
    # Ensure context is always a dict before serializing
    if hasattr(context, "model_dump"):  # Pydantic v2
        context_dict = context.model_dump()
    elif hasattr(context, "dict"):  # Pydantic v1 fallback
        context_dict = context.dict()
    else:
        context_dict = context  # Already a dict
    context_json = json.dumps(context_dict)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversation_state
                (conversation_id, context_json, agent_name, last_response_id)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, context_json, agent_name, last_response_id))
            conn.commit()
            logger.info(
                f"Saved state for conversation_id: {conversation_id}. Agent: {agent_name}, Last response ID: {last_response_id}"
            )
    except sqlite3.Error as e:
        logger.error(f"Error saving state for conversation_id {conversation_id}: {e}", exc_info=True)
        raise
    except Exception as e:
         logger.error(f"Unexpected error saving state for conversation_id {conversation_id}: {e}", exc_info=True)
         raise


def load_state_from_db(conversation_id: str) -> Tuple[SupportTicketContext, str, Optional[str]]:
    """
    Load the context as a raw dict from the DB (no Pydantic validation).
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT context_json, agent_name, last_response_id 
                FROM conversation_state 
                WHERE conversation_id = ?""",
                (conversation_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    context_dict = json.loads(row[0])
                    context = SupportTicketContext(**context_dict)
                except Exception as e:
                    logger.error(f"Failed to load context JSON for {conversation_id}: {e}")
                    raise
                agent_name = row[1]
                last_response_id = row[2]
                logger.info(f"Loaded state for conversation_id: {conversation_id}. Agent: {agent_name}, Last Response ID: {last_response_id}")
                return context, agent_name, last_response_id
            else:
                logger.info(f"No state found for conversation_id: {conversation_id}. Returning default state.")
                return SupportTicketContext(), "", None

    except sqlite3.Error as e:
        logger.error(f"Error loading state for conversation_id {conversation_id} from DB: {e}. Returning default state.", exc_info=True)
        raise
