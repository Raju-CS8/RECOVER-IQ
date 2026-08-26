from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from recoveriq.domain.payment import PaymentStatus
from recoveriq.domain.payment_failure import PaymentFailureCategory
from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.subscription import SubscriptionStatus


@dataclass(frozen=True, slots=True)
class RecoveryState:
    """
    Immutable snapshot of the information available to the recovery
    decision engine at a single point in a recovery episode.

    This object is deliberately a decision-state projection rather than
    a copy of the underlying domain entities. The decision engine should
    receive only information explicitly exposed by this contract.
    """

    customer_id: UUID
    subscription_id: UUID
    payment_id: UUID

    subscription_status: SubscriptionStatus
    payment_status: PaymentStatus

    amount: Decimal
    currency: str

    payment_attempted_at: datetime

    failure_category: PaymentFailureCategory | None
    failure_code: str | None

    recovery_attempt_count: int
    previous_actions: tuple[RecoveryAction, ...]

    available_actions: tuple[RecoveryAction, ...]

    @classmethod
    def create(
        cls,
        *,
        customer_id: UUID,
        subscription_id: UUID,
        payment_id: UUID,
        subscription_status: SubscriptionStatus,
        payment_status: PaymentStatus,
        amount: Decimal,
        currency: str,
        payment_attempted_at: datetime,
        failure_category: PaymentFailureCategory | None = None,
        failure_code: str | None = None,
        recovery_attempt_count: int = 0,
        previous_actions: tuple[RecoveryAction, ...] = (),
        available_actions: tuple[RecoveryAction, ...] = (),
    ) -> RecoveryState:
        """Create a validated recovery-state snapshot."""

        if amount <= Decimal("0"):
            raise ValueError("Recovery state amount must be greater than zero.")

        normalized_currency = currency.strip().upper()

        if not normalized_currency:
            raise ValueError("Recovery state currency must not be empty.")

        if recovery_attempt_count < 0:
            raise ValueError("Recovery attempt count must not be negative.")

        normalized_failure_code = (
            failure_code.strip() if failure_code is not None else None
        )

        if normalized_failure_code == "":
            normalized_failure_code = None

        return cls(
            customer_id=customer_id,
            subscription_id=subscription_id,
            payment_id=payment_id,
            subscription_status=subscription_status,
            payment_status=payment_status,
            amount=amount,
            currency=normalized_currency,
            payment_attempted_at=payment_attempted_at,
            failure_category=failure_category,
            failure_code=normalized_failure_code,
            recovery_attempt_count=recovery_attempt_count,
            previous_actions=tuple(previous_actions),
            available_actions=tuple(available_actions),
        )