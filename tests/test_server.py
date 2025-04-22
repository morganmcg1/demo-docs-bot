# tests/test_server.py
import pytest
from fastapi.testclient import TestClient

# Assuming your FastAPI app instance is named 'app' in server.py
from server import app
from models import SupportTicketContext

# Mark the test to run asynchronously if needed, depends on your test runner setup
@pytest.mark.asyncio
async def test_chat_history_passthrough():
    client = TestClient(app)

    # Update initial_history to use dictionary format
    initial_history = [
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "Hi! How can I help?"} # Assuming 'agent' role for agent messages
    ]
    user_message = "What is W&B?"
    # Pass the list of dicts directly
    initial_context = SupportTicketContext(chat_history=initial_history)

    # Prepare the request payload
    payload = {
        "message": user_message,
        # Pass the model dump which now contains list of dicts
        "context": initial_context.model_dump(),
        "input_items": []
    }

    # Send the request to the endpoint
    response = client.post("/docs-agent", json=payload)

    # Assert the request was successful
    assert response.status_code == 200

    # Parse the response JSON
    response_data = response.json()

    # Adjust assertions if needed based on the new format
    # Assert that the response contains the context and chat_history
    assert "context" in response_data
    assert "chat_history" in response_data["context"]
    returned_history = response_data["context"]["chat_history"]

    # Check length
    assert len(returned_history) > len(initial_history)

    # Check initial items are present (might need adjustment if format changes)
    # This check might be fragile if order changes or content is modified slightly
    assert initial_history[0] in returned_history
    assert initial_history[1] in returned_history

    # Check new user message is present
    assert {"role": "user", "content": user_message} in returned_history

    # Check agent response
    if "answer" in response_data and response_data["answer"]:
        agent_name = list(response_data["answer"].keys())[0] # Still gets 'triage_agent' from 'answer' field
        agent_message = response_data["answer"][agent_name]
        # Check if the agent response dictionary exists in the history WITH 'assistant' role
        assert {"role": "assistant", "content": agent_message} in returned_history

    print("\nReturned Chat History:")
    for item in returned_history:
        print(f"- {item}")
