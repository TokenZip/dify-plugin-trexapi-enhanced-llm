# TrexAPI Plugin — Privacy Policy

**Last updated:** 2026-03-12

## Overview

The TrexAPI plugin enables Dify applications to exchange semantic payloads via the TokenZip Protocol (TZP) and perform RAG (Retrieval-Augmented Generation) retrieval over stored payloads. This privacy policy describes how data is collected, used, and shared.

## Does This Plugin Collect Personal Data?

**No.** This plugin does not directly collect, store, or process any personal data as defined by privacy regulations.

The plugin handles **semantic vector payloads** (quantized embedding vectors, compressed text chunks, and associated metadata) that are transmitted between the user's Dify application and their self-hosted or designated TrexAPI server. These payloads are technical artifacts of AI model processing and do not inherently contain personally identifiable information.

## Data Handling Details

### Data Transmitted

| Data Type | Destination | Purpose |
|-----------|-------------|---------|
| TZP payloads (vectors, compressed text) | User-configured TrexAPI server | Push/pull semantic payloads for cross-agent sharing |
| Search queries (natural language text) | Processed locally by the plugin | RAG retrieval over stored payloads |
| API credentials (TrexAPI key) | User-configured TrexAPI server | Authentication with TrexAPI |

### Data NOT Collected

The plugin does **not** collect or transmit:

- **Type A (Direct Identifiers):** Names, email addresses, phone numbers, government IDs, physical addresses.
- **Type B (Indirect Identifiers):** IP addresses, device identifiers, location data, browsing history, financial information, health information, biometric data.
- **Type C (Combinable Data):** Age, gender, occupation, interests.

### Local Processing

- **Embedding models** (MiniLM, bge-m3): Run entirely on the plugin host machine. No data is sent to external embedding services unless the user explicitly configures Voyage AI (see Third-Party Services below).
- **FAISS vector index**: Built and queried in-memory on the plugin host. No index data is transmitted externally.
- **Cache**: An in-memory LRU cache stores recently accessed payloads for performance. Cache data is never persisted to disk or transmitted.

### Credential Security

- All API keys and credentials are stored securely by the Dify platform using its built-in credential management.
- Credentials are transmitted only to the services they authenticate with (TrexAPI server, downstream LLM API).
- Credentials are never logged, cached in plaintext, or transmitted to any other party.

## Third-Party Services

This plugin may interact with the following third-party services based on user configuration:

### 1. User-Configured TrexAPI Server

- **What is sent:** TZP payloads (push), payload retrieval requests (pull), authentication headers.
- **Controlled by:** The user. The TrexAPI server address is explicitly configured by the user and may be a self-hosted instance or a shared deployment.
- **Privacy policy:** Depends on the TrexAPI deployment. For the reference implementation, see [TrexAPI on GitHub](https://github.com/tokenzip/trexapi).

### 2. Voyage AI (Optional — Model Provider Plugin Only)

- **When used:** Only if the user selects "voyage-3" as the embedding model and provides a Voyage AI API key.
- **What is sent:** Text chunks for embedding computation.
- **Privacy policy:** [Voyage AI Privacy Policy](https://www.voyageai.com/privacy)

### 3. Downstream LLM API (Model Provider Plugin Only)

- **When used:** Only in the TrexAPI Enhanced LLM model provider plugin.
- **What is sent:** Processed prompt messages (with TZP markers resolved to retrieved context) to the user-configured OpenAI-compatible API endpoint.
- **Controlled by:** The user configures the LLM API endpoint and key.
- **Privacy policy:** Depends on the LLM provider chosen by the user (e.g., [OpenAI Privacy Policy](https://openai.com/policies/privacy-policy), [DeepSeek Privacy Policy](https://www.deepseek.com/privacy)).

## Data Retention

This plugin does not persist any data. All processing is ephemeral:

- In-memory caches are cleared when the plugin process stops.
- No data is written to disk by the plugin itself.
- Payload data stored on TrexAPI servers is governed by the TrexAPI server's own retention policies.

## Changes to This Policy

This privacy policy may be updated as the plugin evolves. The "Last updated" date at the top of this document reflects the most recent revision.

## Contact

For privacy-related questions about this plugin, visit [trexapi.com](https://trexapi.com) or open an issue at [GitHub](https://github.com/tokenzip/trexapi/issues).
