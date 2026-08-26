from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.payment import PaymentStatus
from recoveriq.domain.payment_failure import PaymentFailureCategory
from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_state import RecoveryState
from recoveriq.domain.subscription import SubscriptionStatus


@dataclass(frozen=True, slots=True)
class RecoveryFeatureVector:
    """
    Immutable model-facing representation of a recovery decision state.

    RecoveryState is the domain contract. RecoveryFeatureVector is the
    explicit representation exposed to decision models and analytics.

    The feature vector contains no scenario assumptions, probabilities,
    costs, or policy decisions.
    """

    amount: Decimal
    currency: str

    payment_failed: bool
    payment_succeeded: bool
    subscription_active: bool

    failure_category: PaymentFailureCategory | None
    failure_code: str | None

    recovery_attempt_count: int
    previous_action_count: int

    retry_available: bool
    payment_method_update_available: bool
    recovery_message_available: bool
    wait_available: bool
    stop_recovery_available: bool

    @classmethod
    def from_state(
        cls,
        *,
        state: RecoveryState,
    ) -> RecoveryFeatureVector:
        """
        Project a RecoveryState into an explicit feature representation.
        """

        available_actions = set(state.available_actions)

        return cls(
            amount=state.amount,
            currency=state.currency,
            payment_failed=state.payment_status is PaymentStatus.FAILED,
            payment_succeeded=(
                state.payment_status is PaymentStatus.SUCCEEDED
            ),
            subscription_active=(
                state.subscription_status is SubscriptionStatus.ACTIVE
            ),
            failure_category=state.failure_category,
            failure_code=state.failure_code,
            recovery_attempt_count=state.recovery_attempt_count,
            previous_action_count=len(state.previous_actions),
            retry_available=(
                RecoveryAction.RETRY_PAYMENT in available_actions
            ),
            payment_method_update_available=(
                RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE
                in available_actions
            ),
            recovery_message_available=(
                RecoveryAction.SEND_RECOVERY_MESSAGE
                in available_actions
            ),
            wait_available=RecoveryAction.WAIT in available_actions,
            stop_recovery_available=(
                RecoveryAction.STOP_RECOVERY in available_actions
            ),
        )