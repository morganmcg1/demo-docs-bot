"""
Based on openai agents sdk example: https://github.com/openai/openai-agents-python/blob/main/examples/customer_service/main.py
"""

from __future__ import annotations as _annotations

import asyncio
import os
import uuid
import logging
import platform
from typing import Any, Dict, List, Optional, Union, Sequence
from types import SimpleNamespace

import aiohttp
import google.genai as genai
from google import genai
from google.genai import types
from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    trace,
    Model,
    ModelProvider,
    set_tracing_disabled,
    RunConfig,
    Usage,
    ModelResponse,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv
from pydantic import BaseModel

import weave
from gemini_responses import GeminiModelProvider
import httpx

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable OpenAI tracing completely
set_tracing_disabled(True)

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

### CONTEXT

WANDBOT_BASE_URL = "https://wandbot2.replit.app"
WANDBOT_QUERY_ROUTE = "/query"
class WandBAgentContext(BaseModel):
    user_id: str | None = None


### TOOLS


@function_tool
async def query_wandb_support(question: str) -> str:
    """
    Query the Weights & Biases support bot api for help with questions about the
     Weights & Biases platform and how to use W&B Models and W&B Weave.

     W&B features mentioned could include:
     - Experiment tracking with Runs and Sweeps
     - Model management with Models
     - Model management and Data versioning with Artifacts and Registry
     - Collaboration with Teams, Organizations and Reports
     - Visualization with Tables and Charts
     - Tracing and logging with Weave
     - Evaluation and Scorers with Weave Evaluations
     - Weave Datasets

     Responses can take about 20 seconds to generate.

     Args:
         question (str): The question to ask the support bot

     Returns:
         str: The answer to the question
    """

    # Make an actual HTTP request to the W&B support bot API
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{WANDBOT_BASE_URL}{WANDBOT_QUERY_ROUTE}", 
                json={
                    "question": question,
                    "application": "docs_site"
                    }
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    return response_data.get(
                        "answer", "No answer provided by the support bot."
                    )
                else:
                    return (
                        f"Error connecting to W&B support bot: HTTP {response.status}"
                    )
        except Exception as e:
            return f"Error connecting to W&B support bot: {str(e)}"


@function_tool(
    name_override="wandb_support",
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
async def wandb_support(question: str) -> str:
    """
    Query the W&B knowledge bot for support questions.

    Args:
        question: The user's question about W&B.
    Returns:
        The answer from the W&B support bot.
    """
    url = "http://wandbot.replit.app/chat/query"
    payload = {"question": question, "application": "docs-agent"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # The response format is assumed to have an 'answer' field
        return data.get("answer") or str(data)


### AGENTS

# --- OpenAI Model Setup ---
OPENAI_MODEL_NAME = "gpt-4.1-mini-2025-04-14"  # or use "openai/gpt-4o" if desired

# --- AGENT SETUP ---
docs_agent = Agent[WandBAgentContext](
    name="Docs Agent",
    handoff_description="An agent that can support users with their questions about Weights & Biases.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful agent for all W&B questions. Use the provided tool to answer the user's question."
    ),
    tools=[wandb_support],
    model=OPENAI_MODEL_NAME,
)

# --- RUN LOOP ---
from agents import RunConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

async def run_agent_message(user_message: str, input_items: list = None, context: WandBAgentContext = None, current_agent=None, provider=None):
    """
    Run the agent on a single user message. Returns (response_text, updated_input_items, updated_agent).
    """
    if input_items is None:
        input_items = []
    if context is None:
        context = WandBAgentContext()
    if current_agent is None:
        current_agent = docs_agent
    # if provider is None:
    #     provider = ModelProvider()  # Uncomment to use Gemini
    conversation_id = uuid.uuid4().hex[:16]
    input_items.append({"content": user_message, "role": "user"})
    responses = []
    with trace("Docs Agent", group_id=conversation_id):
        result = await Runner.run(current_agent, input_items, context=context)  # provider omitted for OpenAI
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
        input_items = result.to_input_list()
        current_agent = result.last_agent
    return responses, input_items, current_agent

# --- FastAPI server ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set this to your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/docs-agent")
async def run_agent_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    # Optionally support conversation state
    input_items = data.get("input_items", [])
    context_dict = data.get("context", {})
    context = WandBAgentContext(**context_dict) if context_dict else WandBAgentContext()
    responses, updated_input_items, updated_agent = await run_agent_message(
        user_message, input_items=input_items, context=context
    )
    # Get the last assistant message (AI response)
    ai_message = None
    for resp in reversed(responses):
        if resp["agent"] == "Docs Agent":
            ai_message = resp["message"]
            break
    if not ai_message and responses:
        ai_message = responses[-1]["message"]
    return JSONResponse({
        "answer": ai_message or "No response generated."
    })

async def cli_main():
    current_agent: Agent[WandBAgentContext] = docs_agent
    input_items: list[TResponseInputItem] = []
    context = WandBAgentContext()
    # provider = GeminiModelProvider()  # Uncomment to use Gemini
    while True:
        user_input = input("Enter your message: ")
        responses, input_items, current_agent = await run_agent_message(
            user_input,
            input_items=input_items,
            context=context,
            current_agent=current_agent,
            # provider=provider,  # Uncomment to use Gemini
        )
        for resp in responses:
            print(f"{resp['agent']}: {resp['message']}")

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    from dataclasses import dataclass

    @dataclass
    class Args:
        server: bool = False
        port: int = 8000

    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    parsed = parser.parse_args()
    args: Args = parsed.args
    if args.server:
        uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=True)
    else:
        import asyncio
        asyncio.run(cli_main())
