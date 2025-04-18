from __future__ import annotations as _annotations

import asyncio
import logging
import os
import uuid

import simple_parsing
import uvicorn
import weave
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    handoff,
    set_trace_processors,
    trace,
)
from dotenv import load_dotenv
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

from agent_utils import process_agent_step_outputs
from config import DocsAgentConfig
from models import SupportTicketContext
from prompts import (
    SUPPORT_TICKET_AGENT_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
)
from tools import create_ticket, wandbot_support_tool

load_dotenv(override=True)
args: DocsAgentConfig = simple_parsing.parse(DocsAgentConfig)
if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
set_trace_processors([WeaveTracingProcessor()])


### HOOKS


async def on_ticket_created_handoff(
    context: RunContextWrapper[SupportTicketContext],
) -> None:
    # After ticket creation, reset ticket_id if needed or perform any cleanup
    pass


### AGENTS

support_ticket_agent = Agent[SupportTicketContext](
    name="support_ticket_agent",
    model=args.support_ticket_agent_model_name,
    handoff_description="Agent that collects required information to create a support ticket and then creates a support ticket.",
    instructions=SUPPORT_TICKET_AGENT_INSTRUCTIONS,
    tools=[create_ticket],
)

triage_agent = Agent[SupportTicketContext](
    name="triage_agent",
    model=args.triage_agent_model_name,
    handoff_description="A triage agent that can answer general questions about Weights & Biases or hand off to support ticket agent.",
    instructions=TRIAGE_AGENT_INSTRUCTIONS,
    tools=[wandbot_support_tool],
    handoffs=[
        handoff(agent=support_ticket_agent, on_handoff=on_ticket_created_handoff),
    ],
)
support_ticket_agent.handoffs.append(triage_agent)


# Dependency provider function
def get_triage_agent() -> Agent[SupportTicketContext]:
    """Dependency provider for the Triage Agent."""
    # Ensure the agent is initialized before the server starts
    if triage_agent is None:
        raise RuntimeError("Triage agent not initialized")
    return triage_agent


### RUN


async def main(args: DocsAgentConfig):
    current_agent: Agent[SupportTicketContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = SupportTicketContext()

    conversation_id = uuid.uuid4().hex[:16]

    while True:
        user_input = input("Enter your message: ")
        # Maintain chat history in context
        context.chat_history.append(f"user: {user_input}")
        with trace("W&B Docs Agent", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            agent_run = await Runner.run(current_agent, input_items, context=context)

            processed_outputs = process_agent_step_outputs(agent_run)

            for output_data in processed_outputs:
                print(f"{output_data['agent']}: {output_data['message']}")
                if output_data["type"] == "MessageOutputItem":
                    context.chat_history.append(
                        f"{output_data['agent']}: {output_data['message']}"
                    )

            input_items = agent_run.to_input_list()
            current_agent = agent_run.last_agent


if __name__ == "__main__":
    if args.server:
        uvicorn.run(
            "server:app",  # Point to the app instance in server.py
            host="0.0.0.0",
            port=args.port,
            reload=True,
        )
    else:
        asyncio.run(main(args))
