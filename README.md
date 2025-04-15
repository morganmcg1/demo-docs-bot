# Agent for the Weights & Biases Documentation 

A simple chatbot agent that answers questions about Weights & Biases using the OpenAI Agents SDK. The agent can be powered by either OpenAI models or Google Gemini models.

## Setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create a `.env` file based on the `.env.example` template and add your API keys

```bash
cp .env.example .env
# Edit the .env file with your actual API keys
```

3. Run the bot

```bash
python main.py
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required when not using Gemini)
- `GOOGLE_API_KEY`: Your Google API key (required when using Gemini)
- `USE_GEMINI`: Set to "true" to use Google Gemini models instead of OpenAI
- `WANDB_ENTITY`: Your Weights & Biases entity (team or username)
- `WANDB_PROJECT`: Your Weights & Biases project name

## Switching Between OpenAI and Google Gemini

By default, the agent uses OpenAI models. To switch to Google Gemini:

1. Ensure you have set your `GOOGLE_API_KEY` in the `.env` file
2. Set `USE_GEMINI=true` in your `.env` file

The implementation shows how to integrate alternative LLM providers with the OpenAI Agents SDK.