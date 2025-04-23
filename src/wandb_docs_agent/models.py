from pydantic import BaseModel, Field
from typing import Dict, List, Any

class DocsAgentResponse(BaseModel):
    answer: str
    conversation_id: str
    has_error: bool = False
    error_message: str | None = None

class SupportTicketContext(BaseModel):
    user_name: str | None = None
    user_email: str | None = None
    ticket_id: str | None = None
    ticket_name: str | None = None
    ticket_description: str | None = None
    chat_history: Dict[Any, Any] = Field(default_factory=dict)
    # Store the last response ID per agent to maintain separate states
    agent_last_response_ids: Dict[str, str] = Field(default_factory=dict)
    active_agent_name: str | None = None

    class Config:
        extra = "forbid"