from __future__ import annotations

from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_policy_interface import RecoveryPolicy
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata
from recoveriq.domain.recovery_prediction import RecoveryPrediction
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_state import RecoveryState


class RecoveryDecisionPolicy(RecoveryPolicy):
    """
    Deterministic baseline policy for selecting the next recovery action.

    The policy evaluates only actions that are currently available in the
    recovery state.

    For recovery-producing actions, expected value is calculated as:

        expected value =
            (success probability * recoverable amount) - action cost

    This policy is intentionally deterministic and explainable. It provides
    the baseline decision-maker that can later be compared with a learned
    or optimization-based recovery policy.
    """

    POLICY_NAME = "deterministic_baseline"
    POLICY_VERSION = "1.0.0"

    def __init__(self, *, scenario: RecoveryScenario) -> None:
        self._scenario = scenario

    @property
    def metadata(self) -> RecoveryPolicyMetadata:
        """
        Return the stable identity of this policy implementation.
        """

        return RecoveryPolicyMetadata(
            policy_name=self.POLICY_NAME,
            policy_version=self.POLICY_VERSION,
        )

    def choose_action(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryAction:
        """
        Select the highest-value available recovery action.

        If no recovery-producing action has positive expected value,
        WAIT is preferred when available.

        If WAIT is unavailable, STOP_RECOVERY is selected when available.

        If the state contains no viable action at all, ValueError is raised.
        """

        candidates: list[tuple[Decimal, RecoveryAction]] = []

        for action in state.available_actions:
            expected_value = self._expected_value(
                state=state,
                action=action,
            )

            if expected_value > Decimal("0"):
                candidates.append((expected_value, action))

        if candidates:
            candidates.sort(
                key=lambda item: (
                    item[0],
                    self._action_priority(item[1]),
                ),
                reverse=True,
            )

            return candidates[0][1]

        if RecoveryAction.WAIT in state.available_actions:
            return RecoveryAction.WAIT

        if RecoveryAction.STOP_RECOVERY in state.available_actions:
            return RecoveryAction.STOP_RECOVERY

        raise ValueError("No viable recovery action is available.")

    def _predict(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryPrediction:
        """
        Produce an explainable prediction from the deterministic policy.
        """

        action = self.choose_action(
            state=state,
        )

        expected_value = self._expected_value(
            state=state,
            action=action,
        )

        reason = self._decision_reason(
            state=state,
            action=action,
            expected_value=expected_value,
        )

        return RecoveryPrediction(
            action=action,
            expected_value=expected_value,
            reason=reason,
            policy_metadata=self.metadata,
        )

    def _expected_value(
        self,
        *,
        state: RecoveryState,
        action: RecoveryAction,
    ) -> Decimal:
        if action is RecoveryAction.RETRY_PAYMENT:
            return (
                self._scenario.retry_success_probability
                * state.amount
                - self._scenario.retry_cost
            )

        if action is RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE:
            return (
                self._scenario.payment_method_update_success_probability
                * state.amount
                - self._scenario.payment_method_update_cost
            )

        return Decimal("0")

    @staticmethod
    def _action_priority(action: RecoveryAction) -> int:
        """
        Deterministic tie-breaker for recovery-producing actions.

        Retry is preferred over payment-method update when both actions
        have exactly the same expected value.
        """

        priorities = {
            RecoveryAction.RETRY_PAYMENT: 2,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE: 1,
        }

        return priorities.get(action, 0)

    @staticmethod
    def _decision_reason(
        *,
        state: RecoveryState,
        action: RecoveryAction,
        expected_value: Decimal,
    ) -> str:
        if action is RecoveryAction.RETRY_PAYMENT:
            return (
                "Retry payment was selected because it has the highest "
                f"positive expected recovery value of {expected_value}."
            )

        if action is RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE:
            return (
                "Payment-method update was selected because it has the "
                f"highest positive expected recovery value of "
                f"{expected_value}."
            )

        if action is RecoveryAction.WAIT:
            return (
                "Wait was selected because no available recovery-producing "
                "action has positive expected value."
            )

        if action is RecoveryAction.STOP_RECOVERY:
            return (
                "Recovery was stopped because no available recovery-"
                "producing action has positive expected value and waiting "
                "is unavailable."
            )

        return (
            f"Action {action.value} was selected with expected value "
            f"{expected_value}."
        )