from __future__ import annotations

from recoveriq.ai_config import AIProviderConfig
from recoveriq.domain.model_recovery_policy import ModelRecoveryPolicy
from recoveriq.domain.recovery_engine import RecoveryEngine
from recoveriq.domain.recovery_environment import RecoveryEnvironment
from recoveriq.model_provider_factory import RecoveryModelProviderFactory


class RecoveryEngineFactory:
    """
    Application-level factory for constructing a configured recovery engine.

    Infrastructure configuration and provider construction remain outside
    the domain layer.
    """

    @staticmethod
    def create(
        *,
        config: AIProviderConfig,
        environment: RecoveryEnvironment,
    ) -> RecoveryEngine:
        """
        Build a RecoveryEngine using the configured model provider and
        supplied recovery environment.
        """

        provider = RecoveryModelProviderFactory.create(
            config=config,
        )

        policy = ModelRecoveryPolicy(
            provider=provider,
        )

        return RecoveryEngine(
            policy=policy,
            environment=environment,
        )
