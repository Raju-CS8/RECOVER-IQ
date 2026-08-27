from decimal import Decimal

from pydantic import BaseModel, Field

from recoveriq.domain.payment_failure import PaymentFailureCategory


class RecoveryRequest(BaseModel):
    """Request payload for a payment recovery simulation."""

    amount: Decimal = Field(
        default=Decimal("500.00"),
        gt=0,
    )

    failure_category: PaymentFailureCategory = (
        PaymentFailureCategory.TRANSIENT
    )

    retry_success_probability: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        le=1,
    )

    payment_method_update_success_probability: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        le=1,
    )

    retry_cost: Decimal = Field(
        default=Decimal("5.00"),
        ge=0,
    )

    payment_method_update_cost: Decimal = Field(
        default=Decimal("10.00"),
        ge=0,
    )

    recovery_message_cost: Decimal = Field(
        default=Decimal("1.00"),
        ge=0,
    )

    customer_contact_cost: Decimal = Field(
        default=Decimal("2.00"),
        ge=0,
    )

    maximum_recovery_attempts: int = Field(
        default=3,
        ge=0,
    )

    simulator_seed: int = 42

    maximum_steps: int = Field(
        default=3,
        ge=1,
    )


class RecoveryDecisionResponse(BaseModel):
    """Structured API representation of one recovery decision."""

    action: str

    expected_value: Decimal | None
    confidence: Decimal | None
    reason: str | None

    recovered: bool
    recovered_amount: Decimal
    recovery_cost: Decimal

    customer_contacted: bool
    terminal: bool


class RecoveryResponse(BaseModel):
    """Response payload for a completed payment recovery simulation."""

    recovered: bool
    recovered_amount: Decimal

    total_recovery_cost: Decimal
    customer_contact_cost: Decimal
    total_economic_cost: Decimal

    net_recovery_value: Decimal

    decision_count: int
    customer_contact_count: int

    terminal: bool

    decisions: list[RecoveryDecisionResponse]


class BatchRecoveryItemRequest(BaseModel):
    """One failed payment to include in a batch recovery run."""

    amount: Decimal = Field(
        gt=0,
    )

    failure_category: PaymentFailureCategory = (
        PaymentFailureCategory.TRANSIENT
    )


class BatchRecoveryRequest(BaseModel):
    """Request payload for a batch payment recovery simulation."""

    items: list[BatchRecoveryItemRequest] = Field(
        min_length=1,
    )

    retry_success_probability: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        le=1,
    )

    payment_method_update_success_probability: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        le=1,
    )

    retry_cost: Decimal = Field(
        default=Decimal("5.00"),
        ge=0,
    )

    payment_method_update_cost: Decimal = Field(
        default=Decimal("10.00"),
        ge=0,
    )

    recovery_message_cost: Decimal = Field(
        default=Decimal("1.00"),
        ge=0,
    )

    customer_contact_cost: Decimal = Field(
        default=Decimal("2.00"),
        ge=0,
    )

    maximum_recovery_attempts: int = Field(
        default=3,
        ge=0,
    )

    simulator_seed: int = 42

    maximum_steps: int = Field(
        default=3,
        ge=1,
    )


class BatchRecoveryItemResponse(BaseModel):
    """Recovery result for one item in a batch."""

    recovered: bool
    recovered_amount: Decimal

    total_recovery_cost: Decimal
    customer_contact_cost: Decimal
    total_economic_cost: Decimal

    net_recovery_value: Decimal

    decision_count: int
    customer_contact_count: int

    terminal: bool


class BatchRecoveryResponse(BaseModel):
    """Aggregate response for a completed batch recovery run."""

    evaluation_count: int
    recovered_count: int

    recovered_amount: Decimal

    total_recovery_cost: Decimal
    customer_contact_cost: Decimal
    total_economic_cost: Decimal

    net_recovery_value: Decimal

    recovery_rate: Decimal

    items: list[BatchRecoveryItemResponse]