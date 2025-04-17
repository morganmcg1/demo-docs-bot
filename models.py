from pydantic import BaseModel

class SupportTicketContext(BaseModel):
    user_name: str | None = None
    user_email: str | None = None
    ticket_id: str | None = None
    ticket_name: str | None = None
    ticket_description: str | None = None
    chat_history: list[str] = []