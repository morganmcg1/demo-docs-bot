# Agent for the Weights & Biases Documentation 

A simple chatbot agent that answers questions about Weights & Biases using the OpenAI Agents SDK. The agent can be powered by either OpenAI models or Google Gemini models.

## Setup

1. **Install uv (Python package manager)**

If you don't have `uv` installed, install it with:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

2. **Install dependencies with uv**

```bash
uv sync
```

3. **Create a `.env` file based on the `.env.example` template and add your API keys**

```bash
cp .env.example .env
# Edit the .env file with your actual API keys
```

4. **Run the bot using uv**

```bash
uv run python -m wandb_docs_agent.main
```

5. **Or Run the bot server uv**

```bash
uv run python -m wandb_docs_agent.main --server
```

## Required Environment Variables (.env)

Create a `.env` file in the project root with the following variables (see below for which are required for your use case):

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Weights & Biases
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_wandb_project
```

- Only the variables relevant to your provider(s) are strictly required.
- Do not commit your .env file to version control!
