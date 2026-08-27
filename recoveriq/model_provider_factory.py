from __future__ import annotations

from recoveriq.ai_config import AIProviderConfig
from recoveriq.deterministic_model_provider import (
    DeterministicModelProvider,
)
from recoveriq.domain.recovery_model_provider import (
    RecoveryModelProvider,
)


class RecoveryModelProviderFactory:
    """
    Creates the configured recovery model provider.

    Provider selection is centralized here so policies and domain components
    remain independent of infrastructure-specific provider construction.
    """

    @staticmethod
    def create(
        *,
        config: AIProviderConfig,
    ) -> RecoveryModelProvider:
        """
        Create a model provider from the supplied configuration.
        """

        if config.provider == "deterministic":
            return DeterministicModelProvider()

        raise ValueError(
            f"Unsupported AI provider: {config.provider}"
        )