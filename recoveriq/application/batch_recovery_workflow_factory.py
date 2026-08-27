from __future__ import annotations

from recoveriq.ai_config import AIProviderConfig
from recoveriq.domain.batch_recovery_evaluation import (
    BatchRecoveryEvaluator,
)
from recoveriq.domain.batch_recovery_workflow import (
    BatchRecoveryWorkflow,
)
from recoveriq.domain.recovery_episode import RecoveryEpisode
from recoveriq.domain.recovery_environment import RecoveryEnvironment
from recoveriq.domain.recovery_evaluation import RecoveryEvaluator
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_simulator import RecoverySimulator
from recoveriq.recovery_engine_factory import RecoveryEngineFactory


class BatchRecoveryWorkflowFactory:
    """
    Application-level factory for constructing a batch recovery workflow.

    The factory centralizes construction of the shared recovery environment,
    recovery episode, individual evaluator, and batch evaluator.
    """

    @staticmethod
    def create(
        *,
        config: AIProviderConfig,
        scenario: RecoveryScenario,
        simulator_seed: int,
        maximum_steps: int,
    ) -> BatchRecoveryWorkflow:
        """
        Build a fully configured batch recovery workflow.

        The configured recovery episode is reused for every input in the
        batch while each input retains its own recovery state and scenario.
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
        batch_evaluator = BatchRecoveryEvaluator()

        return BatchRecoveryWorkflow(
            episode=episode,
            evaluator=evaluator,
            batch_evaluator=batch_evaluator,
        )
