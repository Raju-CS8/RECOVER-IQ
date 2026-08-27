from __future__ import annotations

from recoveriq.ai_config import AIProviderConfig
from recoveriq.application.recovery_workflow import RecoveryWorkflow
from recoveriq.domain.recovery_environment import RecoveryEnvironment
from recoveriq.domain.recovery_episode import RecoveryEpisode
from recoveriq.domain.recovery_evaluation import RecoveryEvaluator
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_simulator import RecoverySimulator
from recoveriq.recovery_engine_factory import RecoveryEngineFactory


class RecoveryWorkflowFactory:
    """
    Application-level factory for constructing a complete recovery workflow.

    The factory centralizes assembly of the recovery environment, AI-powered
    recovery engine, episode runner, and business evaluator.
    """

    @staticmethod
    def create(
        *,
        config: AIProviderConfig,
        scenario: RecoveryScenario,
        simulator_seed: int,
        maximum_steps: int,
    ) -> RecoveryWorkflow:
        """
        Build a fully configured recovery workflow.

        The supplied scenario controls recovery execution conditions.
        The simulator seed keeps probability-based execution reproducible.
        """

        simulator = RecoverySimulator(
            seed=simulator_seed,
        )

        environment = RecoveryEnvironment(
            scenario=scenario,
            simulator=simulator,
        )

        engine = RecoveryEngineFactory.create(
            config=config,
            environment=environment,
        )

        episode = RecoveryEpisode(
            engine=engine,
            maximum_steps=maximum_steps,
        )

        evaluator = RecoveryEvaluator()

        return RecoveryWorkflow(
            episode=episode,
            evaluator=evaluator,
        )