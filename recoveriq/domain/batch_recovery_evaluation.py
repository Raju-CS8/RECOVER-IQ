from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.recovery_evaluation import RecoveryEvaluation


@dataclass(frozen=True, slots=True)
class BatchRecoveryEvaluation:
    """
    Immutable aggregate evaluation for multiple completed recovery
    evaluations.
    """

    evaluation_count: int
    recovered_count: int

    recovered_amount: Decimal

    total_recovery_cost: Decimal
    customer_contact_cost: Decimal
    total_economic_cost: Decimal

    net_recovery_value: Decimal

    @property
    def recovery_rate(self) -> Decimal:
        """
        Return the proportion of evaluations that successfully recovered
        payment value.
        """

        if self.evaluation_count == 0:
            return Decimal("0")

        return (
            Decimal(self.recovered_count)
            / Decimal(self.evaluation_count)
        )


class BatchRecoveryEvaluator:
    """
    Application-level evaluator that aggregates business metrics across
    multiple recovery evaluations.
    """

    def evaluate(
        self,
        *,
        evaluations: tuple[RecoveryEvaluation, ...],
    ) -> BatchRecoveryEvaluation:
        """
        Aggregate metrics from completed recovery evaluations.
        """

        evaluation_count = len(evaluations)

        recovered_count = sum(
            1
            for evaluation in evaluations
            if evaluation.recovered
        )

        recovered_amount = sum(
            (
                evaluation.recovered_amount
                for evaluation in evaluations
            ),
            Decimal("0"),
        )

        total_recovery_cost = sum(
            (
                evaluation.total_recovery_cost
                for evaluation in evaluations
            ),
            Decimal("0"),
        )

        customer_contact_cost = sum(
            (
                evaluation.customer_contact_cost
                for evaluation in evaluations
            ),
            Decimal("0"),
        )

        total_economic_cost = sum(
            (
                evaluation.total_economic_cost
                for evaluation in evaluations
            ),
            Decimal("0"),
        )

        net_recovery_value = sum(
            (
                evaluation.net_recovery_value
                for evaluation in evaluations
            ),
            Decimal("0"),
        )

        return BatchRecoveryEvaluation(
            evaluation_count=evaluation_count,
            recovered_count=recovered_count,
            recovered_amount=recovered_amount,
            total_recovery_cost=total_recovery_cost,
            customer_contact_cost=customer_contact_cost,
            total_economic_cost=total_economic_cost,
            net_recovery_value=net_recovery_value,
        )