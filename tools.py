import logging
import os
import random

import httpx
import requests
from agents import function_tool
from agents.extensions import RunContextWrapper

from models import SupportTicketContext
from prompts import WANDBOT_DESCRIPTION


@function_tool(
    name_override="wandbot_support_tool", description_override=WANDBOT_DESCRIPTION
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


def set_ticket_context(context, user_name, user_email, ticket_name, ticket_description, ticket_id):
    context.context.user_name = user_name
    context.context.user_email = user_email
    context.context.ticket_name = ticket_name
    context.context.ticket_description = ticket_description
    context.context.ticket_id = ticket_id


@function_tool(
    name_override="create_ticket",
    description_override="Create a support ticket with the provided information.",
)
async def create_ticket(
    context: RunContextWrapper[SupportTicketContext],
    ticket_name: str,
    ticket_description: str,
    user_name: str,
    user_email: str,
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
        ticket_id = f"SIMULATED-{random.randint(1000, 9999)}"
        set_ticket_context(context, user_name, user_email, ticket_name, ticket_description, ticket_id)
        msg = (
            f"[SIMULATED] Support ticket {ticket_id} created for {user_name} (email: {user_email})\n"
            f"Title: {ticket_name}\n"
            f"Description: {ticket_description}\n"
            f"Chat History: {context.context.chat_history}"
        )
        if debug:
            msg += (
                "\n[DEBUG] disable_zendesk flag is set. No Zendesk API call was made."
            )
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
                "comment": {
                    "body": ticket_description
                    + "\n\nW&B Agent Chat History:\n"
                    + "\n".join(context.context.chat_history)
                },
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
            set_ticket_context(context, user_name, user_email, ticket_name, ticket_description, ticket_id)
            msg = (
                f"Zendesk ticket {ticket_id} created for {user_name} (email: {user_email})\n"
                f"Title: {ticket_name}\n"
                f"Description: {ticket_description}\n"
                f"Chat History: {context.context.chat_history}"
            )
            if debug:
                msg += f"\n[DEBUG] Zendesk API response: {new_ticket}"
            return msg
        except Exception as e:
            return f"Failed to create Zendesk ticket: {e}"
    else:
        ticket_id = f"TICKET-{random.randint(1000, 9999)}"
        set_ticket_context(context, user_name, user_email, ticket_name, ticket_description, ticket_id)
        msg = (
            f"Support ticket {ticket_id} created for {user_name} (email: {user_email})\n"
            f"Title: {ticket_name}\n"
            f"Description: {ticket_description}\n"
            f"Chat History: {context.context.chat_history}"
        )
        if debug:
            msg += "\n[DEBUG] USE_ZENDESK is not enabled, simulated ticket only."
        return msg
