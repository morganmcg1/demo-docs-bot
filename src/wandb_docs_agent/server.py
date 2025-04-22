import logging
import uuid
from typing import Dict, Optional, Tuple

from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    trace,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .main import AGENTS
from .models import DocsAgentResponse, SupportTicketContext
from .tools import wandbot_tool  # Import the specific tool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Updated State Store (Using response_id) ---
CONVERSATION_STATE: Dict[str, Tuple[str, Optional[str], SupportTicketContext]] = {}


def load_state(conversation_id: str) -> Tuple[str, Optional[str], SupportTicketContext]:
    """Loads state (agent_name, last_response_id, context) or returns defaults."""
    return CONVERSATION_STATE.get(
        conversation_id,
        (
            "triage_agent",
            None,
            SupportTicketContext(),
        ),  # Default state: no previous response_id
    )


def save_state(
    conversation_id: str,
    agent_name: str,
    last_response_id: Optional[str],
    context: SupportTicketContext,
):
    """Saves the current conversation state (agent_name, last_response_id, context)."""
    CONVERSATION_STATE[conversation_id] = (agent_name, last_response_id, context)
    logger.info(
        f"Saved state for conversation_id: {conversation_id}. Last response ID: {last_response_id}"
    )


# --- End State Store ---


@app.post("/docs-agent")
async def run_agent_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message")
    conversation_id = data.get("conversation_id")  # Can be None

    # Improved logging for debugging agent state and conversation ID
    logger.info(f"[INCOMING REQUEST] conversation_id: {conversation_id}")

    # 1. Validate required input and generate conversation_id if missing
    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="'message' is required.",
        )

    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        logger.info(
            f"No conversation_id provided, generated new one: {conversation_id}"
        )

    # Now conversation_id is guaranteed to be a string

    # 2. Load state from store using the guaranteed conversation_id
    current_agent_name, last_response_id, agent_context = load_state(conversation_id)
    logger.info(
        f"[LOAD STATE] For conversation_id: {conversation_id}, loaded agent: {current_agent_name}, \
last_response_id: {last_response_id}"
    )

    # 3. Get the current agent object
    current_agent: Agent[SupportTicketContext] | None = AGENTS.get(current_agent_name)
    if not current_agent:
        logger.error(
            f"Agent '{current_agent_name}' not found for conversation {conversation_id}. Resetting to triage."
        )
        current_agent_name = "triage_agent"
        current_agent = AGENTS[current_agent_name]
        last_response_id = None  # Reset response ID
        agent_context = SupportTicketContext()  # Reset context

    # 4. Prepare inputs for Runner.run (only current message)
    current_input = [{"role": "user", "content": user_message}]

    # 5. Run the agent turn using Runner.run with response_id
    initial_user_query_snippet = user_message[:20]
    logger.info(
        f"Running agent '{current_agent_name}' for conversation {conversation_id} with last_response_id: {last_response_id}"
    )
    print(f"\n--- Running Agent: {current_agent_name} for Conv: {conversation_id} ---")
    print(f"Current Input: {current_input}")
    print(f"Agent Context (Before): {agent_context}")
    print(f"Using Response ID: {last_response_id}")

    try:
        with trace(
            f"Docs Agent ({current_agent_name}) - '{initial_user_query_snippet}...'",
            group_id=conversation_id,
        ):
            result = await Runner.run(
                current_agent,
                input=current_input,  # Rename 'inputs' to 'input'
                context=agent_context,  # Pass the mutable context
                previous_response_id=last_response_id,  # Use the stored response ID
            )

        # Log the full result object for debugging
        logger.info(f"Full Agent Result: {repr(result)}")

        # 6. Process the result
        new_agent_messages = []
        logger.info(
            f"Agent result.new_items: {[type(item).__name__ for item in result.new_items]}\n\n"
        )
        logger.info(
            f"Agent result.last_agent: {getattr(result.last_agent, 'name', repr(result.last_agent))}\n\n"
        )
        logger.info(
            f"Agent result.last_response_id: {getattr(result, 'last_response_id', None)}\n\n"
        )
        logger.info(
            f"Agent result.raw_responses: {[item for item in result.raw_responses]}\n\n"
        )
        logger.info(f"Agent result.input: {result.input}\n\n")
        for idx, item in enumerate(result.new_items):
            logger.info(
                f"  Item {idx}: type={type(item).__name__}, content={getattr(item, 'content', None)}"
            )
            if isinstance(item, MessageOutputItem):
                text_content = ItemHelpers.text_message_output(item)
                logger.info(f"    MessageOutputItem content: {text_content}")
                if text_content:
                    new_agent_messages.append(
                        {"role": "assistant", "content": text_content}
                    )
            else:
                logger.info(f"    Non-message item: {repr(item)}")

        # If no MessageOutputItem was found, check if we stopped at wandbot_tool
        if not new_agent_messages:
            wandbot_call_id = None
            wandbot_output = None
            for item in result.new_items:
                if (
                    isinstance(item, ToolCallItem)
                    and item.raw_item.name == wandbot_tool.name
                ):
                    wandbot_call_id = item.raw_item.call_id
                    logger.info(
                        f"Found ToolCallItem for {wandbot_tool.name} with call_id: {wandbot_call_id}"
                    )
                # Check ToolCallOutputItem's raw_item dictionary for call_id
                elif (
                    isinstance(item, ToolCallOutputItem)
                    and item.raw_item["call_id"] == wandbot_call_id
                ):
                    wandbot_output = item.output
                    logger.info(
                        f"Found matching ToolCallOutputItem for {wandbot_tool.name} with output."
                    )
                    break  # Found the output, no need to check further

            if wandbot_output:
                logger.info(
                    f"Using output from {wandbot_tool.name} as assistant message."
                )
                new_agent_messages.append(
                    {"role": "assistant", "content": wandbot_output}
                )
            else:
                logger.warning(
                    "No MessageOutputItem and no wandbot_tool output found in result!"
                )

        # 7. Update chat history in the context
        if current_input:
            agent_context.chat_history.extend(current_input)
        if new_agent_messages:
            agent_context.chat_history.extend(new_agent_messages)

        # Defensive agent handoff logic
        handoff_detected = any(
            type(item).__name__ in ["HandoffCallItem", "HandoffOutputItem"]
            for item in result.new_items
        )
        if handoff_detected:
            next_agent_name = result.last_agent.name
        else:
            next_agent_name = (
                current_agent_name  # Stay on the current agent unless explicit handoff
            )

        new_response_id = (
            result.last_response_id
        )  # Use the property to get the last response ID

        logger.info(
            f"Agent run finished. Next agent: {next_agent_name}. New Response ID: {new_response_id}. New messages: {len(new_agent_messages)}"
        )
        logger.info(f"Agent Context (After): {agent_context}")
        logger.info(f"Next Agent: {next_agent_name}")
        logger.info(f"New Response ID: {new_response_id}")
        logger.info(f"New Messages: {new_agent_messages}")
        logger.info("------------------------------------------------------------\n")

        # 8. Save updated state (including the *new* response ID)
        save_state(conversation_id, next_agent_name, new_response_id, agent_context)
        logger.info(
            f"[SAVE STATE] For conversation_id: {conversation_id}, saved agent: {next_agent_name}, new_response_id: {new_response_id}"
        )

        # 9. Prepare and return response to client
        assistant_message_content = "No response"
        agent_error = False
        agent_error_message = None
        if new_agent_messages:
            # Ensure content is string, handle potential non-string tool outputs if necessary
            raw_content = new_agent_messages[0]["content"]
            assistant_message_content = (
                str(raw_content)
                if raw_content is not None
                else "Agent returned empty content."
            )
        else:
            agent_error = True
            agent_error_message = "Agent issue: [No text response generated]"

        return DocsAgentResponse(
            answer=assistant_message_content,
            conversation_id=conversation_id,
            has_error=agent_error,
            error_message=agent_error_message,
        )

    except Exception as e:
        logger.exception(f"Error during agent run for conversation {conversation_id}")
        save_state(conversation_id, current_agent_name, last_response_id, agent_context)
        raise HTTPException(status_code=500, detail=f"Agent processing error: {e}")
