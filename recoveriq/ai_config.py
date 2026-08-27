from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class AIProviderConfig:
    """
    Immutable configuration for an AI recovery provider.

    Configuration is read from the process environment. Secrets are never
    given a default value and are never exposed through repr or logging.
    """

    provider: str
    api_key: str | None
    model: str
    base_url: str | None = None

    @classmethod
    def from_environment(cls) -> AIProviderConfig:
        """
        Build AI provider configuration from environment variables.
        """

        provider = os.getenv("AI_PROVIDER", "").strip()

        api_key = os.getenv("AI_API_KEY")
        if api_key is not None:
            api_key = api_key.strip() or None

        model = os.getenv("AI_MODEL", "").strip()

        base_url = os.getenv("AI_BASE_URL")
        if base_url is not None:
            base_url = base_url.strip() or None

        if not provider:
            raise ValueError(
                "AI_PROVIDER must be configured."
            )

        if not model:
            raise ValueError(
                "AI_MODEL must be configured."
            )

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )