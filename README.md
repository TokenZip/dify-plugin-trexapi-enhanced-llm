# TrexAPI Enhanced LLM — Dify Model Provider Plugin

Wraps any OpenAI-compatible LLM with a TZP (TokenZip Protocol) interceptor. Automatically detects and resolves `[TZP: tx_xx_XXXXXXXXX]` markers in prompts by performing RAG retrieval via TrexAPI, then forwards the enriched prompt to the downstream LLM.

## Prerequisites

- Python >= 3.12
- A running TrexAPI instance (local or remote)
- A Dify platform instance (self-hosted or Dify Cloud)
- An OpenAI-compatible LLM API endpoint

## Required Credentials

| Credential | Where to obtain | Purpose |
|------------|----------------|---------|
| LLM API Base URL | Your LLM provider (e.g. `https://api.openai.com/v1`) | Downstream LLM requests |
| LLM API Key | Your LLM provider's dashboard | Authentication with downstream LLM |
| TrexAPI Base URL | Your TrexAPI deployment (e.g. `http://localhost:3000`) | Fetching TZP payloads |
| TrexAPI API Key | TrexAPI `DEV_API_KEY` or HMAC key | Authentication with TrexAPI |

## How It Works

```
User message (with [TZP: tx_us_xxx])
    → TZP Interceptor detects markers
    → TrexAPI Pull (fetch payload)
    → RAG Retrieval (build FAISS index, search)
    → Inject context into prompt
    → Forward to downstream LLM
    → Return response
```

When no `[TZP: ...]` markers are found, the plugin passes messages directly to the downstream LLM with no overhead.

## Setup

```bash
cd dify_plugin_model
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Dify debug connection details
python -m main
```

## Packaging

```bash
dify plugin package ./dify_plugin_model
```

## Embedding Models

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| `minilm` | 384 | ~80MB | Default, lightweight |
| `bge-m3` | 1024 | ~2GB | Higher quality |

## License

Apache-2.0 OR CC-BY-SA-4.0
