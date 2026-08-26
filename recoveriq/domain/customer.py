from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


class CustomerStatus(str, Enum):
    """Lifecycle status of a customer in the recovery environment."""

    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass(frozen=True, slots=True)
class Customer:
    """
    Immutable representation of a customer in the RecoverIQ environment.

    The customer model intentionally contains only domain information
    required at this stage. Recovery-specific behavior belongs to the
    payment/recovery domain rather than the customer identity itself.
    """

    id: UUID
    created_at: datetime
    status: CustomerStatus

    @classmethod
    def create(
        cls,
        *,
        created_at: datetime | None = None,
        status: CustomerStatus = CustomerStatus.ACTIVE,
    ) -> Customer:
        """Create a new customer with a generated unique identifier."""
        timestamp = created_at or datetime.now(timezone.utc)

        return cls(
            id=uuid4(),
            created_at=timestamp,
            status=status,
        )