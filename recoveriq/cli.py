from __future__ import annotations

import argparse
import sys
from decimal import Decimal, InvalidOperation

from recoveriq.ai_config import AIProviderConfig
from recoveriq.application.recovery_workflow_factory import (
    RecoveryWorkflowFactory,
)
from recoveriq.domain.customer import Customer
from recoveriq.domain.payment import Payment, PaymentStatus
from recoveriq.domain.payment_failure import (
    PaymentFailure,
    PaymentFailureCategory,
)
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.subscription import Subscription


def _parse_amount(value: str) -> Decimal:
    """Parse and validate a positive payment amount."""

    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(
            "Amount must be a valid decimal number."
        ) from error

    if amount <= Decimal("0"):
        raise argparse.ArgumentTypeError(
            "Amount must be greater than zero."
        )

    return amount


def _parse_probability(value: str) -> Decimal:
    """Parse and validate a probability between zero and one."""

    try:
        probability = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(
            "Probability must be a valid decimal number."
        ) from error

    if not Decimal("0") <= probability <= Decimal("1"):
        raise argparse.ArgumentTypeError(
            "Probability must be between 0 and 1."
        )

    return probability


def _parse_positive_integer(value: str) -> int:
    """Parse and validate a positive integer."""

    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Value must be a valid integer."
        ) from error

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def _parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for the RecoverIQ demonstration."""

    parser = argparse.ArgumentParser(
        description="Run a RecoverIQ recovery demonstration."
    )

    parser.add_argument(
        "--failure-category",
        choices=tuple(
            category.value
            for category in PaymentFailureCategory
        ),
        default=PaymentFailureCategory.TRANSIENT.value,
        help=(
            "Payment failure category to simulate. "
            "Defaults to transient."
        ),
    )

    parser.add_argument(
        "--amount",
        type=_parse_amount,
        default=Decimal("500.00"),
        help=(
            "Payment amount to recover. "
            "Defaults to 500.00."
        ),
    )

    parser.add_argument(
        "--retry-success-probability",
        type=_parse_probability,
        default=Decimal("1.0"),
        help=(
            "Probability that a payment retry succeeds. "
            "Must be between 0 and 1. Defaults to 1.0."
        ),
    )

    parser.add_argument(
        "--payment-method-update-success-probability",
        type=_parse_probability,
        default=Decimal("1.0"),
        help=(
            "Probability that a payment method update succeeds. "
            "Must be between 0 and 1. Defaults to 1.0."
        ),
    )

    parser.add_argument(
        "--simulator-seed",
        type=int,
        default=42,
        help=(
            "Seed used for reproducible recovery simulation. "
            "Defaults to 42."
        ),
    )

    parser.add_argument(
        "--maximum-steps",
        type=_parse_positive_integer,
        default=3,
        help=(
            "Maximum number of recovery workflow steps. "
            "Must be greater than zero. Defaults to 3."
        ),
    )

    return parser.parse_args(arguments)


def _create_scenario(
    *,
    failure_category: PaymentFailureCategory,
    retry_success_probability: Decimal,
    payment_method_update_success_probability: Decimal,
) -> RecoveryScenario:
    """Create deterministic scenario conditions for a failure category."""

    return RecoveryScenario(
        failure_category=failure_category,
        retry_success_probability=retry_success_probability,
        payment_method_update_success_probability=(
            payment_method_update_success_probability
        ),
        retry_cost=Decimal("5.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("2.00"),
        maximum_recovery_attempts=3,
    )


def main() -> None:
    """Run a complete deterministic RecoverIQ recovery demonstration."""

    command_arguments = sys.argv[1:]

    if sys.argv[0].endswith("pytest") or sys.argv[0].endswith(
        "__main__.py"
    ):
        command_arguments = []

    arguments = _parse_arguments(command_arguments)

    failure_category = PaymentFailureCategory(
        arguments.failure_category
    )

    amount = arguments.amount

    config = AIProviderConfig.from_environment()

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=amount,
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=amount,
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=failure_category,
        code=f"{failure_category.value}_failure",
    )

    scenario = _create_scenario(
        failure_category=failure_category,
        retry_success_probability=(
            arguments.retry_success_probability
        ),
        payment_method_update_success_probability=(
            arguments.payment_method_update_success_probability
        ),
    )

    workflow = RecoveryWorkflowFactory.create(
        config=config,
        scenario=scenario,
        simulator_seed=arguments.simulator_seed,
        maximum_steps=arguments.maximum_steps,
    )

    result = workflow.run(
        customer=customer,
        subscription=subscription,
        payment=payment,
        failure=failure,
        scenario=scenario,
    )

    print("RecoverIQ Recovery Result")
    print("=" * 30)
    print(
        f"Failure category: "
        f"{failure_category.value}"
    )
    print(f"Payment amount: {amount}")
    print(
        "Retry success probability: "
        f"{arguments.retry_success_probability}"
    )
    print(
        "Payment method update success probability: "
        f"{arguments.payment_method_update_success_probability}"
    )
    print(
        f"Simulator seed: "
        f"{arguments.simulator_seed}"
    )
    print(
        f"Maximum steps: "
        f"{arguments.maximum_steps}"
    )
    print(f"Recovered: {result.evaluation.recovered}")
    print(
        f"Recovered amount: "
        f"{result.evaluation.recovered_amount}"
    )
    print(
        f"Total recovery cost: "
        f"{result.evaluation.total_recovery_cost}"
    )
    print(
        f"Net recovery value: "
        f"{result.evaluation.net_recovery_value}"
    )
    print(
        f"Decision count: "
        f"{result.evaluation.decision_count}"
    )
    print(f"Terminal: {result.evaluation.terminal}")


if __name__ == "__main__":
    main()