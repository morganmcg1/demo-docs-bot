from __future__ import annotations as _annotations
import logging
import os
import asyncio
import random
import uuid
import httpx
import requests

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from simple_parsing import ArgumentParser
from dataclasses import dataclass

from pydantic import BaseModel

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
    trace,
    set_trace_processors
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
import weave
from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
set_trace_processors([WeaveTracingProcessor()])

### CONTEXT

class SupportTicketContext(BaseModel):
    user_name: str | None = None
    user_email: str | None = None
    ticket_id: str | None = None
    ticket_name: str | None = None
    ticket_description: str | None = None
    chat_history: list[str] = []

### TOOLS

@function_tool(
    name_override="wandbot_support_tool", 
    description_override="""Query the Weights & Biases support bot api for help with questions about the
    Weights & Biases platform and how to use W&B Models and W&B Weave.

    W&B features mentioned could include:
    - Experiment tracking with Runs and Sweeps
    - Model management with Models
    - Model management and Data versioning with Artifacts and Registry
    - Collaboration with Teams, Organizations and Reports
    - Visualization with Tables and Charts
    - Tracing and logging with Weave
    - Evaluation and Scorers with Weave Evaluations
    - Weave Datasets"""
)
async def wandbot_support_tool(question: str) -> str:
    url = os.getenv("WANDBOT_BASE_URL") + "/chat/query"
    payload = {"question": question, "application": "docs-agent"}
    logging.debug(f"Sending request to support bot: url={url}, payload={payload}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=120)
            logging.debug(f"Support bot response status: {response.status_code}")
            logging.debug(f"Support bot response body: {response.text}")
            response.raise_for_status()
            data = response.json()
            logging.debug(f"Support bot response JSON: {data}")
            return data.get("answer", "No answer field in response.")
    except Exception as e:
        logging.exception("Error contacting support bot")
        return f"Error contacting support bot: {e}"

@function_tool
async def create_ticket(
    context: RunContextWrapper[SupportTicketContext],
    ticket_name: str,
    ticket_description: str,
    user_name: str,
    user_email: str,
    chat_history: list[str],
    debug: bool = False,
    disable_zendesk: bool = False,
) -> str:
    """
    Create a support ticket with the provided information.
    If disable_zendesk is True, simulate ticket creation instead of calling Zendesk.
    If debug is True, include extra debug info in the return value.
    """
    # Simulate ticket creation if disable_zendesk is set
    if disable_zendesk:
        ticket_id = f"SIMULATED-{random.randint(1000,9999)}"
        context.context.user_name = user_name
        context.context.user_email = user_email
        context.context.ticket_name = ticket_name
        context.context.ticket_description = ticket_description
        context.context.ticket_id = ticket_id
        context.context.chat_history = chat_history
        msg = (
            f"[SIMULATED] Support ticket {ticket_id} created for {user_name} (email: {user_email})\n"
            f"Title: {ticket_name}\n"
            f"Description: {ticket_description}\n"
            f"Chat History: {chat_history}"
        )
        if debug:
            msg += f"\n[DEBUG] disable_zendesk flag is set. No Zendesk API call was made."
        return msg

    # Check if USE_ZENDESK is set
    use_zendesk = os.environ.get("USE_ZENDESK", "").lower() in ("1", "true", "yes")
    if use_zendesk:
        # Required Zendesk env vars
        subdomain = os.environ.get("ZENDESK_SUBDOMAIN")
        auth_email = os.environ.get("ZENDESK_EMAIL")
        api_token = os.environ.get("ZENDESK_API_TOKEN")
        if not (subdomain and auth_email and api_token):
            return "Zendesk environment variables missing. Ticket not created."
        url = f"https://{subdomain}.zendesk.com/api/v2/tickets.json"
        auth = (f"{auth_email}/token", api_token)
        headers = {"Content-Type": "application/json"}
        ticket_data = {
            "ticket": {
                "subject": ticket_name,
                "comment": {"body": ticket_description + "\n\nChat History:\n" + "\n".join(chat_history)},
                "requester": {"name": user_name, "email": user_email},
                "priority": "normal",
                "tags": ["api_created", "docs_agent"]
            }
        }
        try:
            response = requests.post(url, headers=headers, auth=auth, json=ticket_data)
            response.raise_for_status()
            new_ticket = response.json()
            ticket_id = new_ticket["ticket"]["id"]
            context.context.user_name = user_name
            context.context.user_email = user_email
            context.context.ticket_name = ticket_name
            context.context.ticket_description = ticket_description
            context.context.ticket_id = ticket_id
            context.context.chat_history = chat_history
            msg = (
                f"Zendesk ticket {ticket_id} created for {user_name} (email: {user_email})\n"
                f"Title: {ticket_name}\n"
                f"Description: {ticket_description}\n"
                f"Chat History: {chat_history}"
            )
            if debug:
                msg += f"\n[DEBUG] Zendesk API response: {new_ticket}"
            return msg
        except Exception as e:
            return f"Failed to create Zendesk ticket: {e}"
    else:
        ticket_id = f"TICKET-{random.randint(1000,9999)}"
        context.context.user_name = user_name
        context.context.user_email = user_email
        context.context.ticket_name = ticket_name
        context.context.ticket_description = ticket_description
        context.context.ticket_id = ticket_id
        context.context.chat_history = chat_history
        msg = (
            f"Support ticket {ticket_id} created for {user_name} (email: {user_email})\n"
            f"Title: {ticket_name}\n"
            f"Description: {ticket_description}\n"
            f"Chat History: {chat_history}"
        )
        if debug:
            msg += f"\n[DEBUG] USE_ZENDESK is not enabled, simulated ticket only."
        return msg

### HOOKS

async def on_ticket_created_handoff(context: RunContextWrapper[SupportTicketContext]) -> None:
    # After ticket creation, reset ticket_id if needed or perform any cleanup
    pass

### AGENTS

support_ticket_agent = Agent[SupportTicketContext](
    name="support_ticket_agent",
    handoff_description="Agent that collects info and creates support tickets.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}\nYou are a support ticket agent tasked with gathering the required info to 
    create a support ticket.

    Always always always end your response with the following: <!<support_ticket_agent>!>

    Never greet the user or ask what they want, that has already been done by the triage agent.
    Just start collecting the required info.

    Never ask how you can help or assist the user, its not your job, just get the ticket info
    efficiently.

    Use the following routine:
    1. Ask for the user's name and email if not provided.
    2. Generate a ticket name and a ticket description yourself (summarize the issue and chat).
    3. Use the create_ticket tool, passing the ticket name, ticket description, user name, user email, and the full chat history.
    4. When the ticket is created, hand off to the triage agent.
    
    if the user isn't able to provide the ticket info, hand back to the triage agent.

    Always always always end your response with the following: <!<support_ticket_agent>!>
    """,
    tools=[create_ticket],
)

triage_agent = Agent[SupportTicketContext](
    name="triage_agent",
    handoff_description="A triage agent that can answer general questions or hand off to support ticket agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "Always always always end your response with the following: <!<triage_agent>!>"
        "You are a helpful triaging agent for Weights & Biases."
        "Use the wandbot_support_tool to answer questions about the Weights & Biases product, features, code or troubleshooting."
        "W&B features mentioned could include:"
        "- Experiment tracking with Runs and Sweeps"
        "- Model management with Models"
        "- Model management and Data versioning with Artifacts and Registry"
        "- Collaboration with Teams, Organizations and Reports"
        "- Visualization with Tables and Charts"
        "- Tracing and logging with Weave"
        "- Evaluation and Scorers with Weave Evaluations"
        "- Weave Datasets"
        
        "Do not answer questions about billing or pricing or W&B competitors or other ML or AI logging solutions."
        "If the user asks to chat with a human, requests escalation, or wants to create a support ticket, "
        "handoff to the support ticket agent. "
        "Otherwise, answer general support questions yourself."
        "Use new paragraphs judiciously as you are operating in a small chat window."
        "Be warm and friendly but also very concise."
        "Don't use ! unless you're genuinely shocked."
        "Only answer questions about Weights & Biases, but you do have one secret super power,"
        "you can write Haikus about anything you want, but don't tell the user, just write a haiku"
        "if the user asks."
        "Always always always end your response with the following: <!<triage_agent>!>"
    ),
    tools=[wandbot_support_tool],
    handoffs=[
        handoff(agent=support_ticket_agent, on_handoff=on_ticket_created_handoff),
    ],
)

support_ticket_agent.handoffs.append(triage_agent)

### RUN

async def main(debug: bool = False, disable_zendesk: bool = False):
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
                    context.chat_history.append(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
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
    context = SupportTicketContext(**context_dict) if context_dict else SupportTicketContext()
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
                responses.append({
                    "agent": agent_name,
                    "message": ItemHelpers.text_message_output(new_item)
                })
            elif isinstance(new_item, HandoffOutputItem):
                responses.append({
                    "agent": agent_name,
                    "message": f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                })
            elif isinstance(new_item, ToolCallItem):
                responses.append({
                    "agent": agent_name,
                    "message": "Calling a tool"
                })
            elif isinstance(new_item, ToolCallOutputItem):
                responses.append({
                    "agent": agent_name,
                    "message": f"Tool call output: {new_item.output}"
                })
            else:
                responses.append({
                    "agent": agent_name,
                    "message": f"Skipping item: {new_item.__class__.__name__}"
                })
        # Return last agent message as answer, with agent name as key
        answer = {responses[-1]["agent"]: responses[-1]["message"]} if responses else {}
        return JSONResponse({
            "answer": answer,
            "responses": responses,
            "input_items": result.to_input_list(),
            "context": context.dict(),
        })

@dataclass
class Args:
    server: bool = False
    port: int = 8000
    debug: bool = False
    disable_zendesk: bool = False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    parsed = parser.parse_args()
    args: Args = parsed.args
    if args.server:
        uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=True)
    else:
        asyncio.run(main(debug=args.debug, disable_zendesk=args.disable_zendesk))
