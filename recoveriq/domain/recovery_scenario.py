from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.payment_failure import PaymentFailureCategory


@dataclass(frozen=True, slots=True)
class RecoveryScenario:
    """
    Deterministic scenario parameters used by the recovery environment.

    A scenario describes the external conditions under which a recovery
    decision is evaluated. It contains no AI decision logic.
    """

    failure_category: PaymentFailureCategory

    retry_success_probability: Decimal
    payment_method_update_success_probability: Decimal

    retry_cost: Decimal
    payment_method_update_cost: Decimal
    recovery_message_cost: Decimal

    customer_contact_cost: Decimal

    maximum_recovery_attempts: int

    def __post_init__(self) -> None:
        probabilities = (
            self.retry_success_probability,
            self.payment_method_update_success_probability,
        )

        for probability in probabilities:
            if not Decimal("0") <= probability <= Decimal("1"):
                raise ValueError(
                    "Recovery probabilities must be between zero and one."
                )

        costs = (
            self.retry_cost,
            self.payment_method_update_cost,
            self.recovery_message_cost,
            self.customer_contact_cost,
        )

        for cost in costs:
            if cost < Decimal("0"):
                raise ValueError("Recovery costs must not be negative.")

        if self.maximum_recovery_attempts < 0:
            raise ValueError(
                "Maximum recovery attempts must not be negative."
            )