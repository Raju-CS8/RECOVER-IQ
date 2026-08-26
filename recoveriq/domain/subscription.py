from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4


class SubscriptionStatus(str, Enum):
    """Lifecycle status of a subscription."""

    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Subscription:
    """
    Immutable recurring subscription within the RecoverIQ environment.

    The subscription represents the recurring financial obligation that can
    become revenue at risk when a payment attempt fails.
    """

    id: UUID
    customer_id: UUID
    created_at: datetime
    amount: Decimal
    currency: str
    status: SubscriptionStatus

    @classmethod
    def create(
        cls,
        *,
        customer_id: UUID,
        amount: Decimal,
        currency: str = "INR",
        created_at: datetime | None = None,
        status: SubscriptionStatus = SubscriptionStatus.ACTIVE,
    ) -> Subscription:
        """Create a new recurring subscription."""
        if amount <= Decimal("0"):
            raise ValueError("Subscription amount must be greater than zero.")

        normalized_currency = currency.strip().upper()

        if len(normalized_currency) != 3:
            raise ValueError("Currency must be a three-letter ISO-style code.")

        timestamp = created_at or datetime.now(timezone.utc)

        return cls(
            id=uuid4(),
            customer_id=customer_id,
            created_at=timestamp,
            amount=amount,
            currency=normalized_currency,
            status=status,
        )