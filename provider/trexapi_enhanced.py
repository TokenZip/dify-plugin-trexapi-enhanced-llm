"""TrexAPI Enhanced LLM provider — credential validation.

For customizable-model providers, validation happens at the model level,
so the provider-level validate is a pass-through.
"""

from dify_plugin import ModelProvider


class TrexAPIEnhancedProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        pass
