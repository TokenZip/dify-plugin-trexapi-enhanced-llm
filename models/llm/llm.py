"""TZP-Enhanced LLM — wraps an OpenAI-compatible model with TZP interceptor.

Intercepts prompt messages containing [TZP: tx_xx_...] markers,
resolves them via TrexAPI RAG retrieval, injects context, then
forwards the enriched prompt to the downstream LLM.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Optional, Union

import httpx

from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
)
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

from lib.interceptor_sync import TZPInterceptorSync
from lib.models import InterceptorConfig, RAGConfig, TrexConfig
from lib.parser import extract_tzp_tags

logger = logging.getLogger(__name__)


def _build_interceptor(credentials: dict) -> TZPInterceptorSync:
    trex_config = TrexConfig(
        api_base_url=credentials.get("trexapi_base_url", "http://localhost:3000"),
        api_key=credentials.get("trexapi_api_key", ""),
    )
    rag_config = RAGConfig(
        embedding_model=credentials.get("embedding_model", "minilm"),
    )
    config = InterceptorConfig(trex=trex_config, rag=rag_config)
    return TZPInterceptorSync(config)


def _extract_text(msg: PromptMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return ""


def _has_tzp_markers(messages: list[PromptMessage]) -> bool:
    for msg in messages:
        text = _extract_text(msg)
        if text and extract_tzp_tags(text):
            return True
    return False


def _process_messages(
    messages: list[PromptMessage], interceptor: TZPInterceptorSync, user_query: str
) -> list[PromptMessage]:
    """Process all messages through the TZP interceptor."""
    processed: list[PromptMessage] = []
    for msg in messages:
        text = _extract_text(msg)
        if text and extract_tzp_tags(text):
            new_text = interceptor.process(text, user_query)
            if isinstance(msg, SystemPromptMessage):
                processed.append(SystemPromptMessage(content=new_text))
            elif isinstance(msg, UserPromptMessage):
                processed.append(UserPromptMessage(content=new_text))
            elif isinstance(msg, AssistantPromptMessage):
                processed.append(AssistantPromptMessage(content=new_text))
            else:
                processed.append(msg)
        else:
            processed.append(msg)
    return processed


def _get_user_query(messages: list[PromptMessage]) -> str:
    """Extract the last user message as the query for RAG retrieval."""
    for msg in reversed(messages):
        if isinstance(msg, UserPromptMessage):
            text = _extract_text(msg)
            if text:
                return text
    return ""


def _build_openai_messages(messages: list[PromptMessage]) -> list[dict]:
    result = []
    for msg in messages:
        text = _extract_text(msg)
        if isinstance(msg, SystemPromptMessage):
            result.append({"role": "system", "content": text})
        elif isinstance(msg, UserPromptMessage):
            result.append({"role": "user", "content": text})
        elif isinstance(msg, AssistantPromptMessage):
            result.append({"role": "assistant", "content": text})
        else:
            result.append({"role": "user", "content": text})
    return result


class TrexAPIEnhancedLLM(LargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        messages = list(prompt_messages)

        if _has_tzp_markers(messages):
            user_query = _get_user_query(messages)
            interceptor = _build_interceptor(credentials)
            try:
                messages = _process_messages(messages, interceptor, user_query)
            finally:
                interceptor.close()

        api_base = credentials.get("llm_api_base", "https://api.openai.com/v1").rstrip("/")
        api_key = credentials.get("llm_api_key", "")
        openai_messages = _build_openai_messages(messages)

        body: dict = {
            "model": model,
            "messages": openai_messages,
            "stream": stream,
        }
        if model_parameters.get("temperature") is not None:
            body["temperature"] = model_parameters["temperature"]
        if model_parameters.get("top_p") is not None:
            body["top_p"] = model_parameters["top_p"]
        if model_parameters.get("max_tokens") is not None:
            body["max_tokens"] = model_parameters["max_tokens"]
        if stop:
            body["stop"] = stop

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if stream:
            return self._stream_invoke(api_base, headers, body, model)
        else:
            return self._sync_invoke(api_base, headers, body, model, prompt_messages)

    def _sync_invoke(
        self,
        api_base: str,
        headers: dict,
        body: dict,
        model: str,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        try:
            resp = httpx.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise InvokeConnectionError("LLM API request timed out")
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)

        data = resp.json()
        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "")
        usage_data = data.get("usage", {})

        base_usage = LLMUsage.empty_usage()
        usage = LLMUsage(
            **(
                base_usage.model_dump()
                | {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens")
                    or usage_data.get("prompt_tokens", 0)
                    + usage_data.get("completion_tokens", 0),
                }
            )
        )

        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=AssistantPromptMessage(content=content),
            usage=usage,
        )

    def _stream_invoke(
        self,
        api_base: str,
        headers: dict,
        body: dict,
        model: str,
    ) -> Generator[LLMResultChunk, None, None]:
        try:
            with httpx.stream(
                "POST",
                f"{api_base}/chat/completions",
                headers=headers,
                json=body,
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk_data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    usage = LLMUsage.empty_usage()
                    if finish_reason and "usage" in chunk_data:
                        u = chunk_data["usage"]
                        base_usage = LLMUsage.empty_usage()
                        usage = LLMUsage(
                            **(
                                base_usage.model_dump()
                                | {
                                    "prompt_tokens": u.get("prompt_tokens", 0),
                                    "completion_tokens": u.get("completion_tokens", 0),
                                    "total_tokens": u.get("total_tokens")
                                    or u.get("prompt_tokens", 0)
                                    + u.get("completion_tokens", 0),
                                }
                            )
                        )

                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=[],
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=content),
                            finish_reason=finish_reason,
                            usage=usage,
                        ),
                    )
        except httpx.TimeoutException:
            raise InvokeConnectionError("LLM API stream request timed out")
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)

    def _handle_http_error(self, e: httpx.HTTPStatusError) -> None:
        status = e.response.status_code
        detail = e.response.text[:500]
        if status == 401:
            raise InvokeAuthorizationError(f"Authentication failed: {detail}")
        elif status == 429:
            raise InvokeRateLimitError(f"Rate limit exceeded: {detail}")
        elif status >= 500:
            raise InvokeServerUnavailableError(f"Server error {status}: {detail}")
        else:
            raise InvokeBadRequestError(f"Request error {status}: {detail}")

    def validate_credentials(self, model: str, credentials: dict) -> None:
        api_base = credentials.get("llm_api_base", "").rstrip("/")
        api_key = credentials.get("llm_api_key", "")

        if not api_base or not api_key:
            raise CredentialsValidateFailedError("LLM API base URL and API key are required")

        try:
            resp = httpx.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                timeout=30.0,
            )
            if resp.status_code == 401:
                raise CredentialsValidateFailedError("Invalid LLM API key")
            resp.raise_for_status()
        except CredentialsValidateFailedError:
            raise
        except httpx.TimeoutException:
            raise CredentialsValidateFailedError(f"Connection to {api_base} timed out")
        except httpx.ConnectError:
            raise CredentialsValidateFailedError(f"Cannot connect to {api_base}")
        except Exception as e:
            raise CredentialsValidateFailedError(f"Validation failed: {e}")

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        total = 0
        for msg in prompt_messages:
            text = _extract_text(msg)
            total += max(1, len(text) // 4)
        return total

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            features=[],
            model_properties={
                ModelPropertyKey.MODE: "chat",
                ModelPropertyKey.CONTEXT_SIZE: 128000,
            },
            parameter_rules=[],
        )

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeAuthorizationError: [],
            InvokeBadRequestError: [ValueError, json.JSONDecodeError],
            InvokeConnectionError: [httpx.TimeoutException, httpx.ConnectError],
            InvokeRateLimitError: [],
            InvokeServerUnavailableError: [httpx.HTTPStatusError],
        }
