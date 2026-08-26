from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from recoveriq.domain.customer import Customer, CustomerStatus
from recoveriq.domain.subscription import Subscription, SubscriptionStatus


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


def test_subscription_create_generates_valid_subscription():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="inr",
    )

    assert isinstance(subscription.id, UUID)
    assert subscription.customer_id == customer.id
    assert subscription.amount == Decimal("499.00")
    assert subscription.currency == "INR"
    assert subscription.status is SubscriptionStatus.ACTIVE


def test_subscription_rejects_non_positive_amount():
    customer = Customer.create()

    try:
        Subscription.create(
            customer_id=customer.id,
            amount=Decimal("0"),
        )
    except ValueError as exc:
        assert str(exc) == "Subscription amount must be greater than zero."
    else:
        raise AssertionError("Expected ValueError for non-positive amount.")


def test_subscription_normalizes_currency():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency=" usd ",
    )

    assert subscription.currency == "USD"