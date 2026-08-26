from datetime import datetime, timezone
from uuid import UUID

from recoveriq.domain.customer import Customer, CustomerStatus


def test_recoveriq_package_imports():
    import recoveriq

    assert recoveriq.__name__ == "recoveriq"


def test_customer_create_generates_valid_customer():
    created_at = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)

    customer = Customer.create(created_at=created_at)

    assert isinstance(customer.id, UUID)
    assert customer.created_at == created_at
    assert customer.status is CustomerStatus.ACTIVE


def test_customer_can_be_created_inactive():
    customer = Customer.create(status=CustomerStatus.INACTIVE)

    assert customer.status is CustomerStatus.INACTIVE