from __future__ import annotations

from dataclasses import dataclass

from recoveriq.domain.customer import Customer
from recoveriq.domain.payment import Payment
from recoveriq.domain.payment_failure import PaymentFailure
from recoveriq.domain.recovery_episode import (
    RecoveryEpisode,
    RecoveryEpisodeResult,
)
from recoveriq.domain.recovery_evaluation import (
    RecoveryEvaluation,
    RecoveryEvaluator,
)
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.subscription import Subscription
from recoveriq.recovery_state_factory import RecoveryStateFactory


@dataclass(frozen=True, slots=True)
class RecoveryWorkflowResult:
    """
    Complete result of executing one recovery workflow.
    """

    episode: RecoveryEpisodeResult
    evaluation: RecoveryEvaluation


class RecoveryWorkflow:
    """
    Application-level workflow for one failed-payment recovery process.

    The workflow coordinates state construction, episode execution, and
    business evaluation. Domain decision logic remains inside the domain
    components.
    """

    def __init__(
        self,
        *,
        episode: RecoveryEpisode,
        evaluator: RecoveryEvaluator,
    ) -> None:
        self._episode = episode
        self._evaluator = evaluator

    def run(
        self,
        *,
        customer: Customer,
        subscription: Subscription,
        payment: Payment,
        failure: PaymentFailure,
        scenario: RecoveryScenario,
    ) -> RecoveryWorkflowResult:
        """
        Execute and evaluate one complete recovery workflow.
        """

        state = RecoveryStateFactory.create(
            customer=customer,
            subscription=subscription,
            payment=payment,
            failure=failure,
        )

        episode_result = self._episode.run(
            initial_state=state,
        )

        evaluation = self._evaluator.evaluate(
            episode=episode_result,
            scenario=scenario,
        )

        return RecoveryWorkflowResult(
            episode=episode_result,
            evaluation=evaluation,
        )