
from dataclasses import dataclass

@dataclass
class DocsAgentConfig:
    server: bool = False
    port: int = 8000
    debug: bool = False
    disable_zendesk: bool = False
    triage_agent_model_provider: str = "openai"
    triage_agent_model_name: str = "gpt-4.1-mini-2025-04-15"
    support_ticket_agent_model_provider: str = "openai"
    support_ticket_agent_model_name: str = "gpt-4.1-mini-2025-04-15"
