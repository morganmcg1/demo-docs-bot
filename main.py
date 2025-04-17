from __future__ import annotations as _annotations

import asyncio
import logging
import os
import random
import uuid

import httpx
import requests
import uvicorn
import weave
from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    set_trace_processors,
    trace,
)
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from simple_parsing import simple_parsing
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

from prompts import (
    SUPPORT_TICKET_AGENT_INSTRUCTIONS,
    TRIAGE_AGENT_INSTRUCTIONS,
    WANDBOT_DESCRIPTION,
)
from config import DocsAgentConfig
from models import SupportTicketContext
from support_ticket import create_ticket


logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

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


### RUN


async def main(args: DocsAgentConfig):
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
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(ItemHelpers.text_message_output(new_item))
                    context.chat_history.append(
                        f"{agent_name}: {ItemHelpers.text_message_output(new_item)}"
                    )
                elif isinstance(new_item, HandoffOutputItem):
                    print(
                        f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                    )
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")
                else:
                    print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
            input_items = result.to_input_list()
            current_agent = result.last_agent


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/docs-agent")
async def run_agent_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    input_items = data.get("input_items", [])
    context_dict = data.get("context", {})
    feedback = data.get("feedback", None)
    # Use SupportTicketContext for context
    context = (
        SupportTicketContext(**context_dict) if context_dict else SupportTicketContext()
    )
    # Optionally handle feedback (log or process as needed)
    if feedback is not None:
        logging.info(f"[Feedback received]: {feedback}")
    # Run the agent
    conversation_id = uuid.uuid4().hex[:16]
    input_items.append({"content": user_message, "role": "user"})
    with trace("Docs Agent", group_id=conversation_id):
        result = await Runner.run(triage_agent, input_items, context=context)
        responses = []
        for new_item in result.new_items:
            agent_name = new_item.agent.name
            if isinstance(new_item, MessageOutputItem):
                responses.append(
                    {
                        "agent": agent_name,
                        "message": ItemHelpers.text_message_output(new_item),
                    }
                )
            elif isinstance(new_item, HandoffOutputItem):
                responses.append(
                    {
                        "agent": agent_name,
                        "message": f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}",
                    }
                )
            elif isinstance(new_item, ToolCallItem):
                responses.append({"agent": agent_name, "message": "Calling a tool"})
            elif isinstance(new_item, ToolCallOutputItem):
                responses.append(
                    {
                        "agent": agent_name,
                        "message": f"Tool call output: {new_item.output}",
                    }
                )
            else:
                responses.append(
                    {
                        "agent": agent_name,
                        "message": f"Skipping item: {new_item.__class__.__name__}",
                    }
                )
        # Return last agent message as answer, with agent name as key
        answer = {responses[-1]["agent"]: responses[-1]["message"]} if responses else {}
        return JSONResponse(
            {
                "answer": answer,
                "responses": responses,
                "input_items": result.to_input_list(),
                "context": context.model_dump(),
            }
        )


if __name__ == "__main__":
    args: DocsAgentConfig = simple_parsing.parse(DocsAgentConfig)
    if args.server:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=args.port,
            reload=True,
            factory_args=(args,),
        )
    else:
        asyncio.run(main(args))
