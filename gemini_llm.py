import os
import uuid
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from google import genai
from agents import Usage, ModelResponse, MessageOutputItem

def format_conversation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages to Gemini API format.
    Each message is a dict with 'role' and 'content'.
    Gemini expects a list of dicts with 'role' and 'parts'.
    """
    formatted = []
    for msg in messages:
        # Gemini expects 'user' and 'model' roles only
        role = msg.get("role")
        if role == "assistant":
            role = "model"
        elif role == "system":
            # System messages can be prepended as user message or skipped
            continue
        formatted.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })
    return formatted

class GeminiLLM:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger("GeminiLLM")

    async def generate_response(self, messages: List[Dict[str, Any]], agent=None) -> ModelResponse:
        if not messages:
            error_message = "Sorry, I didn't receive any input to send to Gemini."
            usage_obj = Usage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                requests=1
            )
            output_item = MessageOutputItem(
                raw_item={
                    "role": "assistant",
                    "content": error_message
                },
                agent=agent
            )
            return ModelResponse(
                output=[output_item],
                usage=usage_obj,
                response_id=str(uuid.uuid4())
            )
        try:
            contents = format_conversation(messages)
            if not contents:
                error_message = "Sorry, I couldn't format the conversation for Gemini."
                usage_obj = Usage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    requests=1
                )
                output_item = MessageOutputItem(
                    raw_item={
                        "role": "assistant",
                        "content": error_message
                    },
                    agent=agent
                )
                return ModelResponse(
                    output=[output_item],
                    usage=usage_obj,
                    response_id=str(uuid.uuid4())
                )
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            response_text = response.text if hasattr(response, "text") else str(response)
            usage_metadata = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
            total_tokens = getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0
            usage_obj = Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
                requests=1
            )
            output_item = MessageOutputItem(
                raw_item={
                    "role": "assistant",
                    "content": response_text
                },
                agent=agent
            )
            response_id = getattr(response, "id", str(uuid.uuid4()))
            return ModelResponse(
                output=[output_item],
                usage=usage_obj,
                response_id=response_id
            )
        except Exception as e:
            self.logger.error(f"Error in GeminiLLM.generate_response: {e}")
            error_message = f"Sorry, I encountered an error: {str(e)}"
            usage_obj = Usage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                requests=1
            )
            output_item = MessageOutputItem(
                raw_item={
                    "role": "assistant",
                    "content": error_message
                },
                agent=agent
            )
            return ModelResponse(
                output=[output_item],
                usage=usage_obj,
                response_id=str(uuid.uuid4())
            )
