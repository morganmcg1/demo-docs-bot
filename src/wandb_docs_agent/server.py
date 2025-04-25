import logging
import uuid
import ast
import os
import time
# Remove unused typing imports

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json

from agents import (
    Agent,
    Runner,
    trace,
    ItemHelpers
)
from .models import SupportTicketContext, DocsAgentResponse
from .main import AGENTS
from .database import init_db, load_state_from_db, save_state_to_db
from .utils import map_outputs_to_agents
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# --- OpenAI-compatible request models ---
class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # ... add more fields as needed

class OpenAIChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # ... add more fields as needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Core agent logic, extracted ---
async def run_agent_core(user_message: str, conversation_id: Optional[str], openai_mode: bool = False) -> Dict[str, Any]:
    import uuid, time
    from fastapi import HTTPException
    logger = logging.getLogger(__name__)
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        logger.info(f"No conversation_id provided, generated new one: {conversation_id}")
    agent_context, current_agent_name, last_response_id = load_state_from_db(conversation_id)
    logger.info(f"[LOAD STATE] For conversation_id: {conversation_id}, loaded agent: {current_agent_name}, last_response_id: {last_response_id}")
    current_agent: Agent[SupportTicketContext] | None = AGENTS.get(current_agent_name)
    if not current_agent:
        logger.error(f"Agent '{current_agent_name}' not found for conversation {conversation_id}. Resetting to triage.")
        current_agent_name = "triage_agent"
        current_agent = AGENTS[current_agent_name]
        last_response_id = None
        agent_context = SupportTicketContext()
    current_input = [{"role": "user", "content": user_message}]
    if agent_context.chat_history.get(current_agent_name, None) is not None:
        input_with_history = agent_context.chat_history.get(current_agent_name, []) + current_input
    else:
        input_with_history = current_input
    initial_user_query_snippet = user_message[:20]
    logger.info(f"Running agent '{current_agent_name}' for conversation {conversation_id} with last_response_id: {last_response_id}")
    logger.info(f"\n--- Running Agent: {current_agent_name} for Conv: {conversation_id} ---")
    logger.info(f"Current Input: {current_input}")
    logger.info(f"Running agent '{current_agent.name}' with full input:")
    for inp in input_with_history:
        logger.info(f"  - {inp}")
    logger.info("-----------------\n\n")
    logger.info(f"Agent Context (Before): {agent_context}")
    logger.info(f"Using Response ID: {last_response_id}")
    try:
        with trace(f"Docs Agent ({current_agent_name}) - '{initial_user_query_snippet}...'", group_id=conversation_id):
            result = await Runner.run(
                current_agent,
                input=input_with_history,
                context=agent_context,
                # previous_response_id=last_response_id,
            )

        new_items_input_list = [item.to_input_item() for item in result.new_items]
        current_turn_history = current_input + new_items_input_list
        last_active_agent_name = result.last_agent.name
        logger.info(f"Last active agent name: {last_active_agent_name}")
        last_response_id = result.last_response_id

        logger.info("\n\Current turn history, from new items input list:")
        for item in current_turn_history:
            logger.info(f"  - {item}")
        logger.info("-----------------\n\n")

        logger.info("\n\nNew Items from agent result:")
        logger.info(f"  --- Last Response ID: {last_response_id}\n\n")
        for item in result.new_items:
            logger.info(f"  - Item Type: {item.type}:")
            
            if item.type == "message_output_item":
                logger.info(f"  - Item ID: {item.raw_item.id}")
                logger.info(f"  - Item Text: {item.raw_item.content[:100]}...")
            elif item.type == "tool_call_item":  # ResponseFunctionToolCall
                logger.info(f"  - Item name: {item.raw_item.name}")
                logger.info(f"  - Item Call ID: {item.raw_item.call_id}")
                logger.info(f"  - Item ID: {item.raw_item.id}")
            elif item.type == "tool_call_output_item":
                logger.info(f"  - Item Call ID: {item.raw_item["call_id"]}")
                logger.info(f"  - Item id: {item.raw_item["output"][:150]}...")
            elif item.type == "handoff_call_item":
                logger.info(f"  - Item Name: {item.raw_item.name}")
                logger.info(f"  - Item Call ID: {item.raw_item.call_id}")
                logger.info(f"  - Item id: {item.raw_item.id}...")
            elif item.type == "handoff_output_item":
                logger.info(f"  - Item Call ID: {item.raw_item['call_id']}")
                logger.info(f"  - Item Output: {item.raw_item['output']}")
                output = item.raw_item["output"]
                if isinstance(output, dict):
                    current_agent_tmp = output["assistant"]
                elif isinstance(output, str):
                    try:
                        # Try JSON first (double quotes)
                        current_agent_tmp = json.loads(output)["assistant"]
                    except json.JSONDecodeError:
                        # Fallback: try to eval Python dict string (not recommended for untrusted input)
                        current_agent_tmp = ast.literal_eval(output)["assistant"]
                else:
                    # Handle unexpected types
                    raise ValueError(f"Unsupported output type: {type(output)}")

            else:
                logger.info(f"Need to add logging for type: {item.type}...")
            logger.info("  ------\n\n")

        last_output = result.new_items[-1]
        if last_output.type == "message_output_item":
            final_messages_for_client = ItemHelpers.text_message_output(last_output) + "<!<" + last_active_agent_name+ ">!>"
        elif last_output.type == "tool_call_output_item":
            final_messages_for_client = str(last_output.output) + "<!<" + last_active_agent_name+ ">!>"
        else:
            raise ValueError(f"Error trying to parse `final_messages_for_client`, unexpected last output type: {last_output.type}")
        
        # if last_active_agent_name in agent_context.chat_history:
        #     logger.info("Current chat history (pre update):")
        #     for message in agent_context.chat_history[last_active_agent_name]:
        #         logger.info(f"  {message}")
        #     logger.info("-----------------\n\n")
        
        # Update chat history for all agents in the turn using map_outputs_to_agents
        segmented = map_outputs_to_agents(
            outputs = current_turn_history, 
            initial_agent_name = current_agent.name
        )
        for agent, outputs in segmented.items():
            if agent not in agent_context.chat_history:
                agent_context.chat_history[agent] = []
            # Build set of existing message IDs for this agent
            existing_ids = set()
            for msg in agent_context.chat_history[agent]:
                # Each message may be dict with 'id', or may not have 'id'
                if isinstance(msg, dict) and 'id' in msg:
                    existing_ids.add(msg['id'])
            # Only append outputs with new IDs
            for output in outputs:
                output_id = output.get('id') if isinstance(output, dict) else None
                if output_id is not None:
                    if output_id not in existing_ids:
                        agent_context.chat_history[agent].append(output)
                        existing_ids.add(output_id)
                else:
                    # If no id, fallback: append if not already present as dict
                    if output not in agent_context.chat_history[agent]:
                        agent_context.chat_history[agent].append(output)
        agent_context.active_agent_name = last_active_agent_name
        agent_context.agent_last_response_ids[last_active_agent_name] = last_response_id

        save_state_to_db(
            conversation_id=conversation_id, 
            context=agent_context,
            agent_name=last_active_agent_name, 
            last_response_id=last_response_id
        )

        return {
            "answer": final_messages_for_client,
            "conversation_id": conversation_id,
            "context": agent_context,
            "last_response_id": last_response_id,
            "last_active_agent_name": last_active_agent_name,
        }

    except HTTPException as e:
        logger.error(f"HTTP Exception during agent run for conversation {conversation_id}: {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"Error during agent run for conversation {conversation_id}")
        return {
            "error": str(e),
            "conversation_id": conversation_id,
        }

@app.post("/docs-agent")
async def run_agent_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message")
    conversation_id = data.get("conversation_id")

    logger.info(f"[INCOMING REQUEST] conversation_id: {conversation_id}")

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="'message' is required.",
        )

    result = await run_agent_core(user_message, conversation_id)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return DocsAgentResponse(
        answer=result["answer"],
        conversation_id=result["conversation_id"],
        has_error=False,
        error_message=None,
        )

@app.post("/v1/completions")
async def openai_completions_endpoint(request: Request):
    body = await request.json()
    req = OpenAICompletionRequest(**body)
    # Use prompt as user_message
    user_message = req.prompt
    # Optionally allow conversation_id as a custom field or header, else None
    conversation_id = body.get("conversation_id")
    # Always OpenAI mode for this endpoint
    result = await run_agent_core(user_message, conversation_id, openai_mode=True)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    # OpenAI completions response format
    return {
        "id": f"wandb-{result['conversation_id']}-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": [
            {
                "text": result["answer"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

@app.post("/v1/chat/completions")
async def openai_chat_completions_endpoint(request: Request):
    body = await request.json()
    logger.info(f"[INCOMING REQUEST] BODY: {body}")
    req = OpenAIChatCompletionRequest(**body)
    # Extract the last user message
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found in 'messages'.")
    conversation_id = body.get("conversation_id")
    result = await run_agent_core(user_msg, conversation_id, openai_mode=True)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    # OpenAI chat completions response format
    return {
        "id": f"wandb-{result['conversation_id']}-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["answer"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
