import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    url = "http://127.0.0.1:8000/docs-agent"
    payload = {
        "message": "I need to create a support ticket.",
        "input_items": [],
        "context": {
            "user_name": "Test User",
            "user_email": os.environ.get("ZENDESK_EMAIL", "testuser@example.com"),
            "ticket_name": "Test Ticket from FastAPI Script",
            "ticket_description": "This is a test ticket created via the FastAPI test script.",
            "ticket_id": None,
            "chat_history": [
                "user: Hello, I need help.",
                "bot: Sure, what do you need?",
                "user: I need to create a support ticket.",
                "bot: Could you please provide your name and email address?",
                f"user: My name is Test User and my email is {os.environ.get('ZENDESK_EMAIL', 'testuser@example.com')}",
                "bot: What is the subject of your ticket?",
                "user: The ticket subject is Test Ticket from FastAPI Script.",
                "bot: Please provide a description of your issue.",
                "user: This is a test ticket created via the FastAPI test script."
            ]
        },
        "feedback": None
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        print(f"Status code: {response.status_code}")
        print("Response:")
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
