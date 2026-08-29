from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_state import RecoveryState


@dataclass(frozen=True, slots=True)
class RecoveryOpportunity:
    """
    A candidate recovery action evaluated against a recovery state.

    An opportunity represents a possible business action before the
    recovery environment executes it. It contains estimated value and
    cost information, but does not perform the action.
    """

    action: RecoveryAction
    expected_recovered_amount: Decimal
    expected_recovery_cost: Decimal
    expected_customer_contact_cost: Decimal

    def __post_init__(self) -> None:
        """
        Validate that all estimated economic values are non-negative.
        """

        values = (
            self.expected_recovered_amount,
            self.expected_recovery_cost,
            self.expected_customer_contact_cost,
        )

        if any(value < Decimal("0") for value in values):
            raise ValueError(
                "Recovery opportunity values must not be negative."
            )

    @property
    def expected_total_cost(self) -> Decimal:
        """
        Return the total estimated economic cost of this opportunity.
        """

        return (
            self.expected_recovery_cost
            + self.expected_customer_contact_cost
        )

    @property
    def expected_net_value(self) -> Decimal:
        """
        Return the estimated net economic value of this opportunity.
        """

        return (
            self.expected_recovered_amount
            - self.expected_total_cost
        )

    @classmethod
    def from_state(
        cls,
        *,
        state: RecoveryState,
        action: RecoveryAction,
        expected_recovered_amount: Decimal,
        expected_recovery_cost: Decimal,
        expected_customer_contact_cost: Decimal,
    ) -> RecoveryOpportunity:
        """
        Create a recovery opportunity for an action available in a state.
        """

        if action not in state.available_actions:
            raise ValueError(
                f"Recovery action '{action.value}' is not available "
                "for the current recovery state."
            )

        return cls(
            action=action,
            expected_recovered_amount=expected_recovered_amount,
            expected_recovery_cost=expected_recovery_cost,
            expected_customer_contact_cost=(
                expected_customer_contact_cost
            ),
        )