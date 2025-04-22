import logging
import os
import random

import httpx
import requests
from agents import function_tool
from agents.run_context import RunContextWrapper

from models import SupportTicketContext
from prompts import WANDBOT_TOOL_DESCRIPTION, CREATE_TICKET_TOOL_DESCRIPTION

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(override=True)


@function_tool(description_override=WANDBOT_TOOL_DESCRIPTION)
async def wandbot_tool(question: str) -> str:
    if not os.getenv("WANDBOT_BASE_URL"):
        raise ValueError("WANDBOT_BASE_URL environment variable is not set.")
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


def set_ticket_context(
    context: RunContextWrapper[SupportTicketContext],
    user_name: str,
    user_email: str,
    ticket_name: str,
    ticket_description: str,
    ticket_id: str,
) -> RunContextWrapper[SupportTicketContext]:
    context.context.user_name = user_name
    context.context.user_email = user_email
    context.context.ticket_name = ticket_name
    context.context.ticket_description = ticket_description
    context.context.ticket_id = ticket_id


@function_tool(
    description_override=CREATE_TICKET_TOOL_DESCRIPTION,
)
async def create_ticket(
    context: RunContextWrapper[SupportTicketContext],
    ticket_name: str,
    ticket_description: str,
    user_name: str,
    user_email: str,
) -> str:
    logger.info(
        f"Creating ticket:\nTicket Name: {ticket_name}\nTicket Description: \
{ticket_description}\nUser Name: {user_name}\nUser Email: {user_email}"
    )

    use_zendesk = os.environ.get("USE_ZENDESK", "").lower() in ("1", "true", "yes")
    logger.info(f"Zendesk ticket creation enabled: {use_zendesk}")

    if use_zendesk:
        logger.info("Creating Zendesk ticket.")
        # Required Zendesk env vars
        subdomain = os.environ.get("ZENDESK_SUBDOMAIN")
        auth_email = os.environ.get("ZENDESK_EMAIL")
        api_token = os.environ.get("ZENDESK_API_TOKEN")
        if not (subdomain and auth_email and api_token):
            logger.error("Zendesk environment variables missing. Ticket not created.")
            return "Zendesk environment variables missing. Ticket not created."

        url = f"https://{subdomain}.zendesk.com/api/v2/tickets.json"
        auth = (f"{auth_email}/token", api_token)
        headers = {"Content-Type": "application/json"}

        # Format chat history into a readable string
        chat_history_str = "\n".join(
            [f"[{msg['role'].capitalize()}]: {msg['content']}" for msg in context.context.chat_history]
        )

        # Combine comment body parts into a single string
        comment_body = (
            f"{'='*6} Ticket description from Docs Agent {'='*6}\n\n"
            f"{ticket_description}\n\n"
            f"{'='*6} Chat History {'='*6}\n\n"
            f"{chat_history_str}\n\n"
            f"{'='*25}"
        )

        ticket_data = {
            "ticket": {
                "subject": f"[Docs Agent] {ticket_name}",
                "comment": {"body": comment_body},
                "requester": {"name": user_name, "email": user_email},
                "priority": "normal",
                "tags": ["api_created", "docs_agent"],
            }
        }
        try:
            response = requests.post(url, headers=headers, auth=auth, json=ticket_data)
            response.raise_for_status()
            new_ticket = response.json()
            ticket_id = new_ticket["ticket"]["id"]
            set_ticket_context(
                context,
                user_name,
                user_email,
                ticket_name,
                ticket_description,
                ticket_id,
            )
            msg = (
                f"Zendesk ticket {ticket_id} created for {user_name} (email: {user_email})\n"
                f"Title: {ticket_name}\n"
                f"Description: {ticket_description}\n"
                f"Chat History: {context.context.chat_history}"
            )
            msg += f"\n[DEBUG] Zendesk API response: {new_ticket}"
            logger.info(msg)
            return msg
        except Exception as e:
            logger.error(f"Failed to create Zendesk ticket: {e}")
            return f"Failed to create Zendesk ticket: {e}"
    else:
        logger.info("Zendesk ticket creation disabled. Simulating ticket creation.")
        ticket_id = f"TICKET-{random.randint(1000, 9999)}"
        set_ticket_context(
            context, user_name, user_email, ticket_name, ticket_description, ticket_id
        )
        msg = (
            f"Support ticket {ticket_id} created for {user_name} (email: {user_email})\n"
            f"Title: {ticket_name}\n"
            f"Description: {ticket_description}\n"
            f"Chat History: {context.context.chat_history}"
        )
        logger.info(msg)
        return msg
