import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def test_zendesk_post():
    subdomain = os.environ.get("ZENDESK_SUBDOMAIN")
    auth_email = os.environ.get("ZENDESK_EMAIL")
    api_token = os.environ.get("ZENDESK_API_TOKEN")
    if not (subdomain and auth_email and api_token):
        print("Zendesk environment variables missing. Aborting test.")
        return
    url = f"https://{subdomain}/api/v2/tickets.json"
    auth = (f"{auth_email}/token", api_token)
    headers = {"Content-Type": "application/json"}
    ticket_data = {
        "ticket": {
            "subject": "Direct Zendesk API Test Ticket",
            "comment": {"body": "This ticket was created by a direct POST from test_create_ticket.py."},
            "requester": {"name": "Test User", "email": auth_email},
            "priority": "normal",
            "tags": ["api_created", "direct_post"]
        }
    }
    try:
        response = requests.post(url, headers=headers, auth=auth, json=ticket_data)
        response.raise_for_status()
        created_ticket = response.json()
        print(f"Successfully created ticket! Ticket ID: {created_ticket['ticket']['id']}")
        print(json.dumps(created_ticket, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error creating ticket: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            try:
                print(f"Error Details: {e.response.json()}")
            except Exception:
                print(f"Error Details (non-JSON): {e.response.text}")

def main():
    test_zendesk_post()

if __name__ == "__main__":
    main()
