from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4


class PaymentStatus(str, Enum):
    """Lifecycle status of a payment attempt."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Payment:
    """
    Immutable payment attempt within the RecoverIQ environment.

    A payment belongs to a subscription and represents one attempt to
    collect the subscription amount.
    """

    id: UUID
    subscription_id: UUID
    attempted_at: datetime
    amount: Decimal
    currency: str
    status: PaymentStatus

    @classmethod
    def create(
        cls,
        *,
        subscription_id: UUID,
        amount: Decimal,
        currency: str = "INR",
        attempted_at: datetime | None = None,
        status: PaymentStatus = PaymentStatus.PENDING,
    ) -> Payment:
        """Create a new payment attempt."""
        if amount <= Decimal("0"):
            raise ValueError("Payment amount must be greater than zero.")

        normalized_currency = currency.strip().upper()

        if len(normalized_currency) != 3:
            raise ValueError("Currency must be a three-letter ISO-style code.")

        timestamp = attempted_at or datetime.now(timezone.utc)

        return cls(
            id=uuid4(),
            subscription_id=subscription_id,
            attempted_at=timestamp,
            amount=amount,
            currency=normalized_currency,
            status=status,
        )