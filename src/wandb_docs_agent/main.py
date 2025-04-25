from __future__ import annotations as _annotations

import logging
import os
import sys
from typing import Dict

import uvicorn
import weave
from agents import (
    Agent,
    handoff,
    set_trace_processors,
)
from dotenv import load_dotenv
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

from .config import DocsAgentConfig
from .models import SupportTicketContext
from .prompts import (
    SUPPORT_TICKET_AGENT_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
)
from .tools import create_ticket, wandbot_tool

load_dotenv(override=True)

def get_default_args():
    # Provide defaults for testing/imports
    from .config import DocsAgentConfig
    return DocsAgentConfig()

if __name__ == "__main__":
    import simple_parsing
    args: DocsAgentConfig = simple_parsing.parse(DocsAgentConfig)
else:
    args = get_default_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


os.environ["SERVER_OPENAI_MODE"] = str(args.openai)

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
set_trace_processors([WeaveTracingProcessor()])


### AGENTS

support_ticket_agent = Agent[SupportTicketContext](
    name="support_ticket_agent",
    model=args.support_ticket_agent_model_name,
    instructions=SUPPORT_TICKET_AGENT_INSTRUCTIONS,
    handoff_description="Agent that creates a support ticket in order to put the user in touch with \
the W&B Support team.",
    tools=[create_ticket],
)

triage_agent = Agent[
    SupportTicketContext
](
    name="triage_agent",
    model=args.triage_agent_model_name,
    handoffs=[support_ticket_agent],
    handoff_description="A triage agent that can answer general questions about Weights & Biases or \
and manage a generaal triage conversation with a user",
    instructions=TRIAGE_AGENT_INSTRUCTIONS,
    tools=[wandbot_tool],
    tool_use_behavior={
        "stop_at_tool_names": [wandbot_tool.name]
    },  # directly return the result from the wandbot tool if its called instead of passing to LLM)
)

support_ticket_agent.handoffs.append(handoff(agent=triage_agent))

# --- Setup handoffs after both agents are defined ---

# --- New: Dictionary to map agent names to objects ---
AGENTS: Dict[str, Agent[SupportTicketContext]] = {
    "triage_agent": triage_agent,
    "support_ticket_agent": support_ticket_agent,
}
# --- End New ---


# # Dependency provider function
# def get_triage_agent() -> Agent[SupportTicketContext]:
#     """Dependency provider for the Triage Agent."""
#     # Ensure the agent is initialized before the server starts
#     if "triage_agent" not in AGENTS: # Use the dict now
#         raise RuntimeError("Triage agent not initialized in AGENTS dict")
#     return AGENTS["triage_agent"]


if __name__ == "__main__":
    if args.server:
        uvicorn.run(
            "wandb_docs_agent.server:app",  # Point to the app instance in server.py
            host="0.0.0.0",
            port=args.port,
            reload=True,
        )
