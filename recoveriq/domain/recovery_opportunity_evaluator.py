from __future__ import annotations

from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_opportunity import RecoveryOpportunity
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_state import RecoveryState


class RecoveryOpportunityEvaluator:
    """
    Evaluate available recovery actions as explicit business opportunities.

    The evaluator estimates the expected recovered amount and economic
    costs for each action without executing the recovery action.
    """

    def __init__(
        self,
        *,
        scenario: RecoveryScenario,
    ) -> None:
        self._scenario = scenario

    def evaluate(
        self,
        *,
        state: RecoveryState,
        action: RecoveryAction,
    ) -> RecoveryOpportunity:
        """
        Evaluate one available recovery action.
        """

        expected_recovered_amount = (
            self._expected_recovered_amount(
                state=state,
                action=action,
            )
        )

        expected_recovery_cost = (
            self._expected_recovery_cost(
                action=action,
            )
        )

        expected_customer_contact_cost = (
            self._expected_customer_contact_cost(
                action=action,
            )
        )

        return RecoveryOpportunity.from_state(
            state=state,
            action=action,
            expected_recovered_amount=expected_recovered_amount,
            expected_recovery_cost=expected_recovery_cost,
            expected_customer_contact_cost=(
                expected_customer_contact_cost
            ),
        )

    def evaluate_all(
        self,
        *,
        state: RecoveryState,
    ) -> tuple[RecoveryOpportunity, ...]:
        """
        Evaluate every action currently available in the recovery state.
        """

        return tuple(
            self.evaluate(
                state=state,
                action=action,
            )
            for action in state.available_actions
        )

    def _expected_recovered_amount(
        self,
        *,
        state: RecoveryState,
        action: RecoveryAction,
    ) -> Decimal:
        if action is RecoveryAction.RETRY_PAYMENT:
            return (
                self._scenario.retry_success_probability
                * state.amount
            )

        if action is RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE:
            return (
                self._scenario.payment_method_update_success_probability
                * state.amount
            )

        return Decimal("0")

    def _expected_recovery_cost(
        self,
        *,
        action: RecoveryAction,
    ) -> Decimal:
        if action is RecoveryAction.RETRY_PAYMENT:
            return self._scenario.retry_cost

        if action is RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE:
            return self._scenario.payment_method_update_cost

        if action is RecoveryAction.SEND_RECOVERY_MESSAGE:
            return self._scenario.recovery_message_cost

        return Decimal("0")

    def _expected_customer_contact_cost(
        self,
        *,
        action: RecoveryAction,
    ) -> Decimal:
        if action in {
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            RecoveryAction.SEND_RECOVERY_MESSAGE,
        }:
            return self._scenario.customer_contact_cost

        return Decimal("0")