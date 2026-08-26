from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class PaymentFailureCategory(str, Enum):
    """High-level categories of payment failures."""

    TRANSIENT = "transient"
    PAYMENT_METHOD = "payment_method"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PaymentFailure:
    """
    Immutable diagnostic event describing why a payment attempt failed.

    The category represents the normalized failure class used by the
    recovery environment. The code preserves the machine-readable failure
    reason observed from the payment provider or simulator.
    """

    id: UUID
    payment_id: UUID
    occurred_at: datetime
    category: PaymentFailureCategory
    code: str

    @classmethod
    def create(
        cls,
        *,
        payment_id: UUID,
        category: PaymentFailureCategory,
        code: str,
        occurred_at: datetime | None = None,
    ) -> PaymentFailure:
        """Create a payment-failure event."""
        normalized_code = code.strip()

        if not normalized_code:
            raise ValueError("Payment failure code must not be empty.")

        timestamp = occurred_at or datetime.now(timezone.utc)

        return cls(
            id=uuid4(),
            payment_id=payment_id,
            occurred_at=timestamp,
            category=category,
            code=normalized_code,
        )