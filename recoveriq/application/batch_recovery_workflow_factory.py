from __future__ import annotations

from recoveriq.ai_config import AIProviderConfig
from recoveriq.domain.batch_recovery_evaluation import (
    BatchRecoveryEvaluator,
)
from recoveriq.domain.batch_recovery_workflow import (
    BatchRecoveryEpisodeFactory,
    BatchRecoveryWorkflow,
    RecoveryEpisodeRunner,
)
from recoveriq.domain.recovery_environment import RecoveryEnvironment
from recoveriq.domain.recovery_episode import RecoveryEpisode
from recoveriq.domain.recovery_evaluation import RecoveryEvaluator
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_simulator import RecoverySimulator
from recoveriq.recovery_engine_factory import RecoveryEngineFactory


class ConfiguredBatchRecoveryEpisodeFactory:
    """
    Creates independently configured recovery episodes for batch items.

    Each episode receives the scenario and deterministic simulator seed
    assigned to its specific batch item.
    """

    def __init__(
        self,
        *,
        config: AIProviderConfig,
        maximum_steps: int,
    ) -> None:
        self._config = config
        self._maximum_steps = maximum_steps

    def create_episode(
        self,
        *,
        scenario: RecoveryScenario,
        simulator_seed: int,
    ) -> RecoveryEpisodeRunner:
        """
        Build a recovery episode for one batch item.
        """

        simulator = RecoverySimulator(
            seed=simulator_seed,
        )

        environment = RecoveryEnvironment(
            scenario=scenario,
            simulator=simulator,
        )

        engine = RecoveryEngineFactory.create(
            config=self._config,
            environment=environment,
        )

        return RecoveryEpisode(
            engine=engine,
            maximum_steps=self._maximum_steps,
        )


class BatchRecoveryWorkflowFactory:
    """
    Application-level factory for constructing batch recovery workflows.

    The factory centralizes configuration shared across the batch while
    allowing every batch item to receive its own scenario-aware recovery
    episode.
    """

    @staticmethod
    def create(
        *,
        config: AIProviderConfig,
        simulator_seed: int,
        maximum_steps: int,
    ) -> BatchRecoveryWorkflow:
        """
        Build a fully configured scenario-aware batch recovery workflow.
        """

        episode_factory: BatchRecoveryEpisodeFactory = (
            ConfiguredBatchRecoveryEpisodeFactory(
                config=config,
                maximum_steps=maximum_steps,
            )
        )

        return BatchRecoveryWorkflow(
            episode_factory=episode_factory,
            evaluator=RecoveryEvaluator(),
            batch_evaluator=BatchRecoveryEvaluator(),
            simulator_seed=simulator_seed,
        )