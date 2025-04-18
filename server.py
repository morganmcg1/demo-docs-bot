import logging
import uuid

from agents import (
    Agent,
    Runner,
    trace,
)
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_utils import process_agent_step_outputs
from main import get_triage_agent
from models import SupportTicketContext

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/docs-agent")
async def run_agent_endpoint(
    request: Request,
    injected_triage_agent: Agent[SupportTicketContext] = Depends(get_triage_agent),
):
    data = await request.json()
    user_message = data.get("message", "")
    input_items = data.get("input_items", [])
    if not isinstance(input_items, list):
        input_items = []

    context_dict = data.get("context", {})
    feedback = data.get("feedback", None)
    context = (
        SupportTicketContext(**context_dict) if context_dict else SupportTicketContext()
    )
    if feedback is not None:
        logging.info(f"[Feedback received]: {feedback}")

    input_items.append({"content": user_message, "role": "user"})

    conversation_id = uuid.uuid4().hex[:16]

    with trace("Docs Agent", group_id=conversation_id):
        docs_agent_app = await Runner.run(
            injected_triage_agent, input_items, context=context
        )
        responses = process_agent_step_outputs(docs_agent_app)
        answer = {responses[-1]["agent"]: responses[-1]["message"]} if responses else {}
        return JSONResponse(
            {
                "answer": answer,
                "responses": responses,
                "input_items": getattr(docs_agent_app, "to_input_list", lambda: [])(),
                "context": getattr(context, "model_dump", lambda: {})(),
            }
        )
