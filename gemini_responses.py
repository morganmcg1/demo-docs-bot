import uuid
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections.abc import AsyncIterator
from agents.usage import Usage
from agents.items import ModelResponse
from agents.models.interface import Model, ModelProvider, ModelTracing
from google import genai
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

if TYPE_CHECKING:
    from agents.agent_output import AgentOutputSchema
    from agents.tool import Tool
    from agents.handoffs import Handoff
    from agents.model_settings import ModelSettings

logger = logging.getLogger("GeminiModel")

class GeminiModel(Model):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    async def get_response(
        self,
        system_instructions: Optional[str],
        input: str | List[Any],
        model_settings: Any,
        tools: Optional[List[Any]],
        output_schema: Any,
        handoffs: Optional[List[Any]],
        tracing: ModelTracing,
        previous_response_id: Optional[str] = None,
    ) -> ModelResponse:
        try:
            messages = input if isinstance(input, list) else [{"role": "user", "content": input}]
            contents = self._format_conversation(messages)
            if not contents:
                raise ValueError("No valid messages to send to Gemini.")
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            response_text = self._extract_text(response)
            usage_obj = Usage(
                input_tokens=getattr(response, "prompt_token_count", 0),
                output_tokens=getattr(response, "candidates_token_count", 0),
                total_tokens=getattr(response, "total_token_count", 0),
                requests=1
            )
            msg = ResponseOutputMessage(
                id=str(uuid.uuid4()),
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text=response_text,
                        type="output_text"
                    )
                ],
                role="assistant",
                status="completed",
                type="message"
            )
            response_id = getattr(response, "id", str(uuid.uuid4()))
            return ModelResponse(
                output=[msg],
                usage=usage_obj,
                response_id=response_id
            )
        except Exception as e:
            logger.error(f"Error in GeminiModel.get_response: {e}")
            error_message = f"Sorry, I encountered an error: {str(e)}"
            usage_obj = Usage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                requests=1
            )
            msg = ResponseOutputMessage(
                id=str(uuid.uuid4()),
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text=error_message,
                        type="output_text"
                    )
                ],
                role="assistant",
                status="completed",
                type="message"
            )
            return ModelResponse(
                output=[msg],
                usage=usage_obj,
                response_id=str(uuid.uuid4())
            )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | List[Any],
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        previous_response_id: Optional[str] = None,
    ) -> AsyncIterator[ResponseOutputMessage]:
        resp = await self.get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
        )
        for item in resp.output:
            yield item

    def _format_conversation(self, messages: List[Any]) -> List[dict]:
        contents = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                if getattr(msg, "role", None) in ("assistant", "model"):
                    # Concatenate all text from content parts
                    text = ""
                    for part in getattr(msg, "content", []):
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str):
                            text += part_text
                    contents.append({
                        "role": "model",
                        "parts": [{"text": text}]
                    })
                elif getattr(msg, "role", None) == "user":
                    user_text = ""
                    if isinstance(msg.content, list):
                        for part in msg.content:
                            part_text = getattr(part, "text", None)
                            if isinstance(part_text, str):
                                user_text += part_text
                    elif isinstance(msg.content, str):
                        user_text = msg.content
                    contents.append({
                        "role": "user",
                        "parts": [{"text": user_text}]
                    })
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = msg["role"]
                content = msg["content"]
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": str(content)}]
                })
            elif isinstance(msg, str):
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg}]
                })
        return contents

    def _extract_text(self, response: Any) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                return part.text
                    elif hasattr(candidate.content, "text") and candidate.content.text:
                        return candidate.content.text
        return "Hello! I'm the Gemini-powered agent. How can I help you?"

class GeminiModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        if model_name and model_name.startswith("gemini-"):
            return GeminiModel(model=model_name)
        raise ValueError(f"GeminiModelProvider only supports gemini-* models, got: {model_name}")
