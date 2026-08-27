from recoveriq.cli import main
from recoveriq.ai_config import AIProviderConfig
from recoveriq.recovery_prediction_parser import RecoveryPredictionParser
from recoveriq.model_provider_factory import RecoveryModelProviderFactory
from recoveriq.recovery_engine_factory import RecoveryEngineFactory
from recoveriq.ai_config import AIProviderConfig
from recoveriq.model_provider_factory import RecoveryModelProviderFactory
from recoveriq.recovery_engine_factory import RecoveryEngineFactory
from recoveriq.domain.recovery_features import RecoveryFeatureVector
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID
from recoveriq.recovery_state_factory import RecoveryStateFactory

from recoveriq.domain.recovery_evaluation import (
    RecoveryEvaluation,
    RecoveryEvaluator,
)
from recoveriq.domain.batch_recovery_evaluation import (
    BatchRecoveryEvaluation,
    BatchRecoveryEvaluator,
)
from recoveriq.application.recovery_workflow import (
    RecoveryWorkflow,
    RecoveryWorkflowResult,
)
from recoveriq.application.recovery_workflow_factory import (
    RecoveryWorkflowFactory,
)
from recoveriq.domain.batch_recovery_runner import (
    BatchRecoveryResult,
    BatchRecoveryRunner,
)
from recoveriq.domain.batch_recovery_workflow import (
    BatchRecoveryInput,
    BatchRecoveryWorkflow,
    BatchRecoveryWorkflowResult,
)
from recoveriq.domain.batch_recovery_runner import (
    BatchRecoveryResult,
    BatchRecoveryRunner,
)
from recoveriq.domain.batch_recovery_evaluation import (
    BatchRecoveryEvaluation,
    BatchRecoveryEvaluator,
)
from recoveriq.domain.recovery_policy_metadata import (
    RecoveryPolicyMetadata,
)
from recoveriq.domain.recovery_feature_encoder import (
    EncodedRecoveryFeatures,
    RecoveryFeatureEncoder,
)
from recoveriq.domain.recovery_policy_interface import RecoveryPolicy
from recoveriq.domain.recovery_prediction import RecoveryPrediction
from recoveriq.domain.recovery_model_provider import RecoveryModelProvider
from recoveriq.domain.model_recovery_policy import ModelRecoveryPolicy
from recoveriq.deterministic_model_provider import DeterministicModelProvider
from recoveriq.domain.customer import Customer, CustomerStatus
from recoveriq.domain.payment import Payment, PaymentStatus
from recoveriq.domain.payment_failure import (
    PaymentFailure,
    PaymentFailureCategory,
)
from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_engine import (
    RecoveryDecision,
    RecoveryEngine,
)
from recoveriq.domain.recovery_environment import (
    RecoveryEnvironment,
    RecoveryOutcome,
)
from recoveriq.domain.recovery_episode import (
    RecoveryEpisode,
    RecoveryEpisodeResult,
)
from recoveriq.domain.recovery_policy import RecoveryDecisionPolicy
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_simulator import (
    RecoverySimulator,
    SimulationResult,
)
from recoveriq.domain.recovery_state import RecoveryState
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


def test_payment_create_generates_valid_payment():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="INR",
    )

    attempted_at = datetime(2026, 8, 26, 13, 0, tzinfo=timezone.utc)

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        currency="INR",
        attempted_at=attempted_at,
    )

    assert isinstance(payment.id, UUID)
    assert payment.subscription_id == subscription.id
    assert payment.attempted_at == attempted_at
    assert payment.amount == Decimal("499.00")
    assert payment.currency == "INR"
    assert payment.status is PaymentStatus.PENDING


def test_payment_can_be_created_as_succeeded():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("999.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("999.00"),
        status=PaymentStatus.SUCCEEDED,
    )

    assert payment.status is PaymentStatus.SUCCEEDED


def test_payment_rejects_non_positive_amount():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    try:
        Payment.create(
            subscription_id=subscription.id,
            amount=Decimal("0"),
        )
    except ValueError as exc:
        assert str(exc) == "Payment amount must be greater than zero."
    else:
        raise AssertionError("Expected ValueError for non-positive amount.")


def test_payment_normalizes_currency():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency=" usd ",
    )

    assert payment.currency == "USD"


def test_payment_failure_create_generates_valid_failure():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    occurred_at = datetime(2026, 8, 26, 13, 5, tzinfo=timezone.utc)

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=PaymentFailureCategory.TRANSIENT,
        code="timeout",
        occurred_at=occurred_at,
    )

    assert isinstance(failure.id, UUID)
    assert failure.payment_id == payment.id
    assert failure.occurred_at == occurred_at
    assert failure.category is PaymentFailureCategory.TRANSIENT
    assert failure.code == "timeout"


def test_payment_failure_normalizes_code():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=PaymentFailureCategory.PAYMENT_METHOD,
        code="  card_declined  ",
    )

    assert failure.code == "card_declined"


def test_payment_failure_rejects_empty_code():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    try:
        PaymentFailure.create(
            payment_id=payment.id,
            category=PaymentFailureCategory.UNKNOWN,
            code="   ",
        )
    except ValueError as exc:
        assert str(exc) == "Payment failure code must not be empty."
    else:
        raise AssertionError("Expected ValueError for an empty failure code.")


def test_recovery_action_space_is_closed():
    assert {action.value for action in RecoveryAction} == {
        "retry_payment",
        "request_payment_method_update",
        "send_recovery_message",
        "wait",
        "stop_recovery",
    }


def test_recovery_state_create_generates_valid_snapshot():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="INR",
    )

    attempted_at = datetime(2026, 8, 26, 14, 0, tzinfo=timezone.utc)

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        currency="INR",
        attempted_at=attempted_at,
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        recovery_attempt_count=1,
        previous_actions=(RecoveryAction.RETRY_PAYMENT,),
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    assert state.customer_id == customer.id
    assert state.subscription_id == subscription.id
    assert state.payment_id == payment.id
    assert state.subscription_status is SubscriptionStatus.ACTIVE
    assert state.payment_status is PaymentStatus.FAILED
    assert state.amount == Decimal("499.00")
    assert state.currency == "INR"
    assert state.payment_attempted_at == attempted_at
    assert state.failure_category is PaymentFailureCategory.TRANSIENT
    assert state.failure_code == "timeout"
    assert state.recovery_attempt_count == 1
    assert state.previous_actions == (RecoveryAction.RETRY_PAYMENT,)
    assert state.available_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT,
        RecoveryAction.STOP_RECOVERY,
    )


def test_recovery_state_normalizes_currency_and_failure_code():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=PaymentStatus.FAILED,
        amount=payment.amount,
        currency=" usd ",
        payment_attempted_at=payment.attempted_at,
        failure_code="  timeout  ",
    )

    assert state.currency == "USD"
    assert state.failure_code == "timeout"


def test_recovery_state_rejects_non_positive_amount():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
    )

    try:
        RecoveryState.create(
            customer_id=customer.id,
            subscription_id=subscription.id,
            payment_id=payment.id,
            subscription_status=subscription.status,
            payment_status=payment.status,
            amount=Decimal("0"),
            currency="INR",
            payment_attempted_at=payment.attempted_at,
        )
    except ValueError as exc:
        assert str(exc) == "Recovery state amount must be greater than zero."
    else:
        raise AssertionError(
            "Expected ValueError for non-positive recovery-state amount."
        )


def test_recovery_state_rejects_negative_attempt_count():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
    )

    try:
        RecoveryState.create(
            customer_id=customer.id,
            subscription_id=subscription.id,
            payment_id=payment.id,
            subscription_status=subscription.status,
            payment_status=payment.status,
            amount=payment.amount,
            currency="INR",
            payment_attempted_at=payment.attempted_at,
            recovery_attempt_count=-1,
        )
    except ValueError as exc:
        assert str(exc) == "Recovery attempt count must not be negative."
    else:
        raise AssertionError(
            "Expected ValueError for negative recovery attempt count."
        )


def test_recovery_state_converts_action_sequences_to_immutable_tuples():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency="INR",
        payment_attempted_at=payment.attempted_at,
        previous_actions=[RecoveryAction.WAIT],
        available_actions=[RecoveryAction.RETRY_PAYMENT],
    )

    assert state.previous_actions == (RecoveryAction.WAIT,)
    assert state.available_actions == (RecoveryAction.RETRY_PAYMENT,)


def test_recovery_scenario_accepts_valid_parameters():
    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.70"),
        payment_method_update_success_probability=Decimal("0.80"),
        retry_cost=Decimal("0.50"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.10"),
        customer_contact_cost=Decimal("0.05"),
        maximum_recovery_attempts=3,
    )

    assert scenario.failure_category is PaymentFailureCategory.TRANSIENT
    assert scenario.retry_success_probability == Decimal("0.70")
    assert scenario.payment_method_update_success_probability == Decimal("0.80")
    assert scenario.retry_cost == Decimal("0.50")
    assert scenario.payment_method_update_cost == Decimal("1.00")
    assert scenario.recovery_message_cost == Decimal("0.10")
    assert scenario.customer_contact_cost == Decimal("0.05")
    assert scenario.maximum_recovery_attempts == 3


def test_recovery_scenario_rejects_probability_above_one():
    try:
        RecoveryScenario(
            failure_category=PaymentFailureCategory.TRANSIENT,
            retry_success_probability=Decimal("1.01"),
            payment_method_update_success_probability=Decimal("0.80"),
            retry_cost=Decimal("0.50"),
            payment_method_update_cost=Decimal("1.00"),
            recovery_message_cost=Decimal("0.10"),
            customer_contact_cost=Decimal("0.05"),
            maximum_recovery_attempts=3,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Recovery probabilities must be between zero and one."
        )
    else:
        raise AssertionError(
            "Expected ValueError for probability above one."
        )


def test_recovery_scenario_rejects_negative_cost():
    try:
        RecoveryScenario(
            failure_category=PaymentFailureCategory.PAYMENT_METHOD,
            retry_success_probability=Decimal("0.20"),
            payment_method_update_success_probability=Decimal("0.80"),
            retry_cost=Decimal("-0.01"),
            payment_method_update_cost=Decimal("1.00"),
            recovery_message_cost=Decimal("0.10"),
            customer_contact_cost=Decimal("0.05"),
            maximum_recovery_attempts=3,
        )
    except ValueError as exc:
        assert str(exc) == "Recovery costs must not be negative."
    else:
        raise AssertionError("Expected ValueError for negative cost.")


def test_recovery_scenario_rejects_negative_maximum_attempts():
    try:
        RecoveryScenario(
            failure_category=PaymentFailureCategory.UNKNOWN,
            retry_success_probability=Decimal("0.20"),
            payment_method_update_success_probability=Decimal("0.80"),
            retry_cost=Decimal("0.50"),
            payment_method_update_cost=Decimal("1.00"),
            recovery_message_cost=Decimal("0.10"),
            customer_contact_cost=Decimal("0.05"),
            maximum_recovery_attempts=-1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Maximum recovery attempts must not be negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError for negative maximum attempts."
        )


def test_recovery_environment_rejects_unavailable_action():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        available_actions=(RecoveryAction.WAIT,),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.70"),
        payment_method_update_success_probability=Decimal("0.80"),
        retry_cost=Decimal("0.50"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.10"),
        customer_contact_cost=Decimal("0.05"),
        maximum_recovery_attempts=3,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    try:
        environment.execute(
            state=state,
            action=RecoveryAction.RETRY_PAYMENT,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Recovery action 'retry_payment' is not available "
            "for the current recovery state."
        )
    else:
        raise AssertionError(
            "Expected ValueError for unavailable recovery action."
        )


def test_recovery_environment_stop_recovery_is_terminal():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        available_actions=(RecoveryAction.STOP_RECOVERY,),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.70"),
        payment_method_update_success_probability=Decimal("0.80"),
        retry_cost=Decimal("0.50"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.10"),
        customer_contact_cost=Decimal("0.05"),
        maximum_recovery_attempts=3,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    outcome = environment.execute(
        state=state,
        action=RecoveryAction.STOP_RECOVERY,
    )

    assert isinstance(outcome, RecoveryOutcome)
    assert outcome.action is RecoveryAction.STOP_RECOVERY
    assert outcome.payment_status is PaymentStatus.FAILED
    assert outcome.recovered is False
    assert outcome.recovered_amount == Decimal("0")
    assert outcome.recovery_cost == Decimal("0")
    assert outcome.customer_contacted is False
    assert outcome.terminal is True


def test_recovery_simulator_returns_simulation_result():
    simulator = RecoverySimulator(seed=42)

    result = simulator.trial(
        success_probability=Decimal("1.0"),
    )

    assert isinstance(result, SimulationResult)
    assert result.successful is True
    assert Decimal("0") <= result.sampled_value < Decimal("1")


def test_recovery_simulator_is_reproducible_for_same_seed():
    first = RecoverySimulator(seed=123)
    second = RecoverySimulator(seed=123)

    first_result = first.trial(
        success_probability=Decimal("0.5"),
    )
    second_result = second.trial(
        success_probability=Decimal("0.5"),
    )

    assert first_result == second_result


def test_recovery_simulator_probability_zero_never_succeeds():
    simulator = RecoverySimulator(seed=7)

    results = [
        simulator.trial(success_probability=Decimal("0.0"))
        for _ in range(10)
    ]

    assert all(result.successful is False for result in results)


def test_recovery_simulator_probability_one_always_succeeds():
    simulator = RecoverySimulator(seed=7)

    results = [
        simulator.trial(success_probability=Decimal("1.0"))
        for _ in range(10)
    ]

    assert all(result.successful is True for result in results)


def test_recovery_simulator_rejects_invalid_probability():
    simulator = RecoverySimulator(seed=42)

    try:
        simulator.trial(success_probability=Decimal("1.01"))
    except ValueError as exc:
        assert str(exc) == (
            "Success probability must be between zero and one."
        )
    else:
        raise AssertionError(
            "Expected ValueError for probability above one."
        )


def test_recovery_environment_successful_retry_recovers_payment_amount():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        recovery_attempt_count=0,
        available_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("0.50"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.10"),
        customer_contact_cost=Decimal("0.05"),
        maximum_recovery_attempts=3,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    outcome = environment.execute(
        state=state,
        action=RecoveryAction.RETRY_PAYMENT,
    )

    assert outcome.action is RecoveryAction.RETRY_PAYMENT
    assert outcome.payment_status is PaymentStatus.SUCCEEDED
    assert outcome.recovered is True
    assert outcome.recovered_amount == Decimal("499.00")
    assert outcome.recovery_cost == Decimal("0.50")
    assert outcome.customer_contacted is False
    assert outcome.terminal is True


def test_recovery_environment_failed_retry_recovers_nothing():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        recovery_attempt_count=0,
        available_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("0.50"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.10"),
        customer_contact_cost=Decimal("0.05"),
        maximum_recovery_attempts=3,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    outcome = environment.execute(
        state=state,
        action=RecoveryAction.RETRY_PAYMENT,
    )

    assert outcome.action is RecoveryAction.RETRY_PAYMENT
    assert outcome.payment_status is PaymentStatus.FAILED
    assert outcome.recovered is False
    assert outcome.recovered_amount == Decimal("0")
    assert outcome.recovery_cost == Decimal("0.50")
    assert outcome.customer_contacted is False
    assert outcome.terminal is False


def test_recovery_policy_selects_retry_when_retry_has_highest_expected_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.80"),
        payment_method_update_success_probability=Decimal("0.40"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("20.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(scenario=scenario)

    action = policy.choose_action(state=state)

    assert action is RecoveryAction.RETRY_PAYMENT


def test_recovery_policy_selects_payment_method_update_when_it_has_higher_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        retry_success_probability=Decimal("0.30"),
        payment_method_update_success_probability=Decimal("0.90"),
        retry_cost=Decimal("20.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(scenario=scenario)

    action = policy.choose_action(state=state)

    assert action is RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE


def test_recovery_policy_prefers_wait_when_no_recovery_action_has_positive_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.01"),
        payment_method_update_success_probability=Decimal("0.01"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(scenario=scenario)

    action = policy.choose_action(state=state)

    assert action is RecoveryAction.WAIT


def test_recovery_policy_selects_stop_when_no_recovery_action_or_wait_exists():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.UNKNOWN,
        failure_code="unknown",
        available_actions=(RecoveryAction.STOP_RECOVERY,),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.UNKNOWN,
        retry_success_probability=Decimal("0.00"),
        payment_method_update_success_probability=Decimal("0.00"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(scenario=scenario)

    action = policy.choose_action(state=state)

    assert action is RecoveryAction.STOP_RECOVERY


def test_recovery_policy_rejects_state_with_no_available_actions():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.UNKNOWN,
        failure_code="unknown",
        available_actions=(),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.UNKNOWN,
        retry_success_probability=Decimal("0.00"),
        payment_method_update_success_probability=Decimal("0.00"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(scenario=scenario)

    try:
        policy.choose_action(state=state)
    except ValueError as exc:
        assert str(exc) == "No viable recovery action is available."
    else:
        raise AssertionError(
            "Expected ValueError when no recovery actions are available."
        )


def test_recovery_engine_runs_policy_and_environment_together():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("20.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    decision = engine.run_once(
        state=state,
    )

    assert isinstance(decision, RecoveryDecision)
    assert decision.action is RecoveryAction.RETRY_PAYMENT
    assert decision.outcome.action is RecoveryAction.RETRY_PAYMENT
    assert decision.outcome.payment_status is PaymentStatus.SUCCEEDED
    assert decision.outcome.recovered is True
    assert decision.outcome.recovered_amount == Decimal("1000.00")


def test_recovery_engine_preserves_wait_decision_when_recovery_value_is_negative():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.01"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("50.00"),
        payment_method_update_cost=Decimal("50.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    decision = engine.run_once(
        state=state,
    )

    assert isinstance(decision, RecoveryDecision)
    assert decision.action is RecoveryAction.WAIT
    assert decision.outcome.action is RecoveryAction.WAIT
    assert decision.outcome.recovered is False
    assert decision.outcome.recovered_amount == Decimal("0")


def test_recovery_episode_stops_after_successful_recovery():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("20.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=5,
    )

    result = episode.run(
        initial_state=state,
    )

    assert isinstance(result, RecoveryEpisodeResult)
    assert result.decision_count == 1
    assert result.recovered is True
    assert result.recovered_amount == Decimal("1000.00")
    assert result.total_recovery_cost == Decimal("10.00")
    assert result.terminal is True
    assert result.final_state.payment_status is PaymentStatus.SUCCEEDED
    assert result.final_state.recovery_attempt_count == 1
    assert result.final_state.previous_actions == (
        RecoveryAction.RETRY_PAYMENT,
    )
    assert result.final_state.available_actions == ()


def test_recovery_episode_tracks_multiple_non_terminal_steps():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.02"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("2.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=2,
    )

    result = episode.run(
        initial_state=state,
    )

    assert result.decision_count == 2
    assert result.recovered is False
    assert result.recovered_amount == Decimal("0")
    assert result.total_recovery_cost == Decimal("1.00")
    assert result.terminal is False
    assert result.final_state.payment_status is PaymentStatus.FAILED
    assert result.final_state.recovery_attempt_count == 1
    assert result.final_state.previous_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT,
    )


def test_recovery_episode_respects_step_limit():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_status=payment.status,
        payment_id=payment.id,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("2.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=1,
    )

    result = episode.run(
        initial_state=state,
    )

    assert result.decision_count == 1
    assert result.terminal is False

def test_recovery_episode_rejects_non_positive_step_limit():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.UNKNOWN,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    try:
        RecoveryEpisode(
            engine=engine,
            maximum_steps=0,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Maximum episode steps must be greater than zero."
        )
    else:
        raise AssertionError(
            "Expected ValueError for non-positive episode step limit."
        )


def test_recovery_evaluator_calculates_net_recovery_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(RecoveryAction.RETRY_PAYMENT,),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("20.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("5.00"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=3,
    )

    episode_result = episode.run(
        initial_state=state,
    )

    evaluator = RecoveryEvaluator()

    evaluation = evaluator.evaluate(
        episode=episode_result,
        scenario=scenario,
    )

    assert isinstance(evaluation, RecoveryEvaluation)
    assert evaluation.recovered is True
    assert evaluation.recovered_amount == Decimal("1000.00")
    assert evaluation.total_recovery_cost == Decimal("10.00")
    assert evaluation.customer_contact_cost == Decimal("0")
    assert evaluation.net_recovery_value == Decimal("990.00")
    assert evaluation.decision_count == 1


def test_recovery_evaluator_includes_customer_contact_cost():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        available_actions=(
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("1.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("20.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("5.00"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=3,
    )

    episode_result = episode.run(
        initial_state=state,
    )

    evaluator = RecoveryEvaluator()

    evaluation = evaluator.evaluate(
        episode=episode_result,
        scenario=scenario,
    )

    assert evaluation.recovered is True
    assert evaluation.recovered_amount == Decimal("500.00")
    assert evaluation.total_recovery_cost == Decimal("20.00")
    assert evaluation.customer_contact_cost == Decimal("5.00")
    assert evaluation.net_recovery_value == Decimal("475.00")
    assert evaluation.decision_count == 1


def test_recovery_evaluator_handles_unsuccessful_episode():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.01"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("0.50"),
        payment_method_update_cost=Decimal("5.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("3.00"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=policy,
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=1,
    )

    episode_result = episode.run(
        initial_state=state,
    )

    evaluator = RecoveryEvaluator()

    evaluation = evaluator.evaluate(
        episode=episode_result,
        scenario=scenario,
    )

    assert evaluation.recovered is False
    assert evaluation.recovered_amount == Decimal("0")
    assert evaluation.total_recovery_cost == Decimal("0.50")
    assert evaluation.customer_contact_cost == Decimal("0")
    assert evaluation.net_recovery_value == Decimal("-0.50")
    assert evaluation.decision_count == 1


def test_recovery_evaluator_counts_multiple_customer_contacts():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        available_actions=(
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        ),
    )

    first_outcome = RecoveryOutcome(
        action=RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        payment_status=PaymentStatus.FAILED,
        recovered=False,
        recovered_amount=Decimal("0"),
        recovery_cost=Decimal("5.00"),
        customer_contacted=True,
        terminal=False,
    )

    second_outcome = RecoveryOutcome(
        action=RecoveryAction.SEND_RECOVERY_MESSAGE,
        payment_status=PaymentStatus.FAILED,
        recovered=False,
        recovered_amount=Decimal("0"),
        recovery_cost=Decimal("2.00"),
        customer_contacted=True,
        terminal=False,
    )

    episode_result = RecoveryEpisodeResult(
        initial_state=state,
        final_state=state,
        decisions=(
            RecoveryDecision(
                action=RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
                outcome=first_outcome,
            ),
            RecoveryDecision(
                action=RecoveryAction.SEND_RECOVERY_MESSAGE,
                outcome=second_outcome,
            ),
        ),
        recovered=False,
        recovered_amount=Decimal("0"),
        total_recovery_cost=Decimal("7.00"),
        terminal=False,
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("5.00"),
        recovery_message_cost=Decimal("2.00"),
        customer_contact_cost=Decimal("3.00"),
        maximum_recovery_attempts=3,
    )

    evaluator = RecoveryEvaluator()

    evaluation = evaluator.evaluate(
        episode=episode_result,
        scenario=scenario,
    )

    assert evaluation.customer_contact_cost == Decimal("6.00")
    assert evaluation.total_recovery_cost == Decimal("7.00")
    assert evaluation.net_recovery_value == Decimal("-13.00")
    assert evaluation.recovered is False
    assert evaluation.decision_count == 2
def test_recovery_feature_vector_projects_recovery_state():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code=" timeout ",
        recovery_attempt_count=2,
        previous_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            RecoveryAction.WAIT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    assert isinstance(features, RecoveryFeatureVector)
    assert features.amount == Decimal("499.00")
    assert features.currency == "INR"
    assert features.payment_failed is True
    assert features.payment_succeeded is False
    assert features.subscription_active is True
    assert features.failure_category is PaymentFailureCategory.TRANSIENT
    assert features.failure_code == "timeout"
    assert features.recovery_attempt_count == 2
    assert features.previous_action_count == 2
    assert features.retry_available is True
    assert features.payment_method_update_available is True
    assert features.recovery_message_available is False
    assert features.wait_available is True
    assert features.stop_recovery_available is True


def test_recovery_feature_vector_reflects_succeeded_payment():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("999.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("999.00"),
        currency="INR",
        status=PaymentStatus.SUCCEEDED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=None,
        failure_code=None,
        available_actions=(),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    assert features.payment_failed is False
    assert features.payment_succeeded is True
    assert features.subscription_active is True
    assert features.failure_category is None
    assert features.failure_code is None
    assert features.retry_available is False
    assert features.payment_method_update_available is False
    assert features.recovery_message_available is False
    assert features.wait_available is False
    assert features.stop_recovery_available is False


def test_recovery_feature_vector_tracks_available_actions():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("250.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("250.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        available_actions=(
            RecoveryAction.SEND_RECOVERY_MESSAGE,
            RecoveryAction.WAIT,
        ),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    assert features.retry_available is False
    assert features.payment_method_update_available is False
    assert features.recovery_message_available is True
    assert features.wait_available is True
    assert features.stop_recovery_available is False
def test_recovery_feature_encoder_produces_stable_numeric_vector():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        recovery_attempt_count=2,
        previous_actions=(RecoveryAction.RETRY_PAYMENT,),
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    encoder = RecoveryFeatureEncoder()

    encoded = encoder.encode(
        features=features,
    )

    assert isinstance(encoded, EncodedRecoveryFeatures)

    assert encoded.values == (
        Decimal("499.00"),
        Decimal("1"),
        Decimal("0"),
        Decimal("1"),
        Decimal("1"),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("2"),
        Decimal("1"),
        Decimal("1"),
        Decimal("0"),
        Decimal("0"),
        Decimal("1"),
        Decimal("0"),
    )


def test_recovery_feature_encoder_exposes_stable_feature_names():
    encoder = RecoveryFeatureEncoder()

    assert encoder.feature_names() == (
        "amount",
        "payment_failed",
        "payment_succeeded",
        "subscription_active",
        "failure_transient",
        "failure_payment_method",
        "failure_authentication",
        "failure_unknown",
        "recovery_attempt_count",
        "previous_action_count",
        "retry_available",
        "payment_method_update_available",
        "recovery_message_available",
        "wait_available",
        "stop_recovery_available",
    )


def test_recovery_feature_encoder_encodes_payment_method_failure():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("250.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("250.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        available_actions=(),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    encoded = RecoveryFeatureEncoder().encode(
        features=features,
    )

    assert encoded.values[4:8] == (
        Decimal("0"),
        Decimal("1"),
        Decimal("0"),
        Decimal("0"),
    )


def test_recovery_feature_encoder_encodes_unknown_failure():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.UNKNOWN,
        failure_code="provider_error",
        available_actions=(),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    encoded = RecoveryFeatureEncoder().encode(
        features=features,
    )

    assert encoded.values[4:8] == (
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        Decimal("1"),
    )
def test_recovery_decision_policy_implements_recovery_policy_contract():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("2.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    assert isinstance(policy, RecoveryPolicy)
    assert policy.choose_action(state=state) is RecoveryAction.RETRY_PAYMENT


def test_recovery_engine_accepts_custom_recovery_policy():
    class FixedRecoveryPolicy(RecoveryPolicy):
        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            assert RecoveryAction.STOP_RECOVERY in state.available_actions
            return RecoveryAction.STOP_RECOVERY

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("499.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("499.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.UNKNOWN,
        failure_code="unknown",
        available_actions=(
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.UNKNOWN,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("1.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngine(
        policy=FixedRecoveryPolicy(),
        environment=environment,
    )

    decision = engine.run_once(
        state=state,
    )

    assert decision.action is RecoveryAction.STOP_RECOVERY
    assert decision.outcome.terminal is True
def test_recovery_prediction_creates_action_only_prediction():
    prediction = RecoveryPrediction(
        action=RecoveryAction.RETRY_PAYMENT,
    )

    assert prediction.action is RecoveryAction.RETRY_PAYMENT
    assert prediction.confidence is None
    assert prediction.expected_value is None
    assert prediction.reason is None
    assert prediction.has_confidence is False
    assert prediction.has_expected_value is False
    assert prediction.has_reason is False


def test_recovery_prediction_accepts_model_metadata():
    prediction = RecoveryPrediction(
        action=RecoveryAction.RETRY_PAYMENT,
        confidence=Decimal("0.85"),
        expected_value=Decimal("420.15"),
        reason="Transient failure with positive recovery value.",
    )

    assert prediction.action is RecoveryAction.RETRY_PAYMENT
    assert prediction.confidence == Decimal("0.85")
    assert prediction.expected_value == Decimal("420.15")
    assert (
        prediction.reason
        == "Transient failure with positive recovery value."
    )
    assert prediction.has_confidence is True
    assert prediction.has_expected_value is True
    assert prediction.has_reason is True


def test_recovery_prediction_rejects_confidence_above_one():
    try:
        RecoveryPrediction(
            action=RecoveryAction.RETRY_PAYMENT,
            confidence=Decimal("1.01"),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Prediction confidence must be between zero and one."
        )
    else:
        raise AssertionError(
            "Expected ValueError for confidence above one."
        )


def test_recovery_prediction_rejects_confidence_below_zero():
    try:
        RecoveryPrediction(
            action=RecoveryAction.RETRY_PAYMENT,
            confidence=Decimal("-0.01"),
        )
    except ValueError as exc:
        assert str(exc) == (
            "Prediction confidence must be between zero and one."
        )
    else:
        raise AssertionError(
            "Expected ValueError for confidence below zero."
        )


def test_recovery_prediction_rejects_blank_reason():
    try:
        RecoveryPrediction(
            action=RecoveryAction.WAIT,
            reason="   ",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Prediction reason must not be empty."
        )
    else:
        raise AssertionError(
            "Expected ValueError for blank prediction reason."
        )
def test_recovery_policy_default_predict_adapts_choose_action():
    class FixedRecoveryPolicy(RecoveryPolicy):
        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            assert RecoveryAction.WAIT in state.available_actions
            return RecoveryAction.WAIT

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    prediction = FixedRecoveryPolicy().predict(
        state=state,
    )

    assert isinstance(prediction, RecoveryPrediction)
    assert prediction.action is RecoveryAction.WAIT
    assert prediction.confidence is None
    assert prediction.expected_value is None
    assert prediction.reason is None


def test_recovery_decision_policy_predict_returns_selected_action():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("2.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    prediction = policy.predict(
        state=state,
    )

    assert isinstance(prediction, RecoveryPrediction)
    assert prediction.action is RecoveryAction.RETRY_PAYMENT
def test_recovery_decision_policy_prediction_contains_expected_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.80"),
        payment_method_update_success_probability=Decimal("0.00"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("5.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    prediction = policy.predict(
        state=state,
    )

    assert prediction.action is RecoveryAction.RETRY_PAYMENT
    assert prediction.expected_value == Decimal("390.00")
    assert prediction.confidence is None
    assert prediction.reason is not None
    assert "positive expected recovery value" in prediction.reason


def test_recovery_decision_policy_wait_prediction_has_zero_expected_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.00"),
        payment_method_update_success_probability=Decimal("0.00"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    prediction = policy.predict(
        state=state,
    )

    assert prediction.action is RecoveryAction.WAIT
    assert prediction.expected_value == Decimal("0")
    assert prediction.confidence is None
    assert prediction.reason == (
        "Wait was selected because no available recovery-producing "
        "action has positive expected value."
    )
def test_recovery_engine_preserves_policy_prediction():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
        ),
    )

    class PredictingPolicy(RecoveryPolicy):
        def predict(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryPrediction:
            return RecoveryPrediction(
                action=RecoveryAction.RETRY_PAYMENT,
                confidence=Decimal("0.90"),
                expected_value=Decimal("400.00"),
                reason="High-value transient recovery.",
            )

        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            return RecoveryAction.RETRY_PAYMENT

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("2.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    engine = RecoveryEngine(
        policy=PredictingPolicy(),
        environment=RecoveryEnvironment(
            scenario=scenario,
            simulator=RecoverySimulator(seed=42),
        ),
    )

    decision = engine.run_once(
        state=state,
    )

    assert decision.action is RecoveryAction.RETRY_PAYMENT
    assert decision.prediction.action is RecoveryAction.RETRY_PAYMENT
    assert decision.prediction.confidence == Decimal("0.90")
    assert decision.prediction.expected_value == Decimal("400.00")
    assert decision.prediction.reason == "High-value transient recovery."

    assert decision.outcome.recovered is True
    assert decision.outcome.payment_status is PaymentStatus.SUCCEEDED


def test_recovery_decision_action_property_matches_prediction():
    prediction = RecoveryPrediction(
        action=RecoveryAction.WAIT,
        confidence=Decimal("0.60"),
        reason="Waiting is preferred.",
    )

    outcome = RecoveryOutcome(
        action=RecoveryAction.WAIT,
        payment_status=PaymentStatus.FAILED,
        recovered=False,
        recovered_amount=Decimal("0"),
        recovery_cost=Decimal("0"),
        customer_contacted=False,
        terminal=False,
    )

    decision = RecoveryDecision(
        prediction=prediction,
        outcome=outcome,
    )

    assert decision.action is RecoveryAction.WAIT
    assert decision.action is decision.prediction.action
def test_recovery_policy_rejects_prediction_for_unavailable_action():
    class InvalidPredictionPolicy(RecoveryPolicy):
        def _predict(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryPrediction:
            return RecoveryPrediction(
                action=RecoveryAction.RETRY_PAYMENT,
            )

        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            return RecoveryAction.RETRY_PAYMENT

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    try:
        InvalidPredictionPolicy().predict(
            state=state,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Predicted recovery action 'retry_payment' "
            "is not available for the current recovery state."
        )
    else:
        raise AssertionError(
            "Expected ValueError for unavailable predicted action."
        )


def test_recovery_policy_accepts_prediction_for_available_action():
    class ValidPredictionPolicy(RecoveryPolicy):
        def _predict(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryPrediction:
            return RecoveryPrediction(
                action=RecoveryAction.WAIT,
                confidence=Decimal("0.75"),
                reason="Waiting is currently available.",
            )

        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            return RecoveryAction.WAIT

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    prediction = ValidPredictionPolicy().predict(
        state=state,
    )

    assert prediction.action is RecoveryAction.WAIT
    assert prediction.confidence == Decimal("0.75")
    assert prediction.reason == "Waiting is currently available."
def test_recovery_policy_metadata_creates_valid_metadata():
    metadata = RecoveryPolicyMetadata(
        policy_name="deterministic_baseline",
        policy_version="1.0.0",
    )

    assert metadata.policy_name == "deterministic_baseline"
    assert metadata.policy_version == "1.0.0"


def test_recovery_policy_metadata_rejects_empty_name():
    try:
        RecoveryPolicyMetadata(
            policy_name="   ",
            policy_version="1.0.0",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Recovery policy name must not be empty."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty policy name."
        )


def test_recovery_policy_metadata_rejects_empty_version():
    try:
        RecoveryPolicyMetadata(
            policy_name="deterministic_baseline",
            policy_version="   ",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Recovery policy version must not be empty."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty policy version."
        )
def test_recovery_prediction_can_include_policy_metadata():
    metadata = RecoveryPolicyMetadata(
        policy_name="deterministic_baseline",
        policy_version="1.0.0",
    )

    prediction = RecoveryPrediction(
        action=RecoveryAction.RETRY_PAYMENT,
        confidence=Decimal("0.85"),
        expected_value=Decimal("420.00"),
        reason="Positive expected recovery value.",
        policy_metadata=metadata,
    )

    assert prediction.policy_metadata == metadata
    assert prediction.has_policy_metadata is True


def test_recovery_prediction_without_policy_metadata_remains_valid():
    prediction = RecoveryPrediction(
        action=RecoveryAction.WAIT,
    )

    assert prediction.policy_metadata is None
    assert prediction.has_policy_metadata is False
def test_recovery_decision_policy_exposes_stable_metadata():
    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.80"),
        payment_method_update_success_probability=Decimal("0.70"),
        retry_cost=Decimal("1.00"),
        payment_method_update_cost=Decimal("2.00"),
        recovery_message_cost=Decimal("0.50"),
        customer_contact_cost=Decimal("0.25"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    metadata = policy.metadata

    assert metadata.policy_name == "deterministic_baseline"
    assert metadata.policy_version == "1.0.0"


def test_recovery_decision_policy_prediction_contains_policy_metadata():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("0.80"),
        payment_method_update_success_probability=Decimal("0.00"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("5.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("0.50"),
        maximum_recovery_attempts=3,
    )

    policy = RecoveryDecisionPolicy(
        scenario=scenario,
    )

    prediction = policy.predict(
        state=state,
    )

    assert prediction.policy_metadata is not None
    assert prediction.policy_metadata.policy_name == (
        "deterministic_baseline"
    )
    assert prediction.policy_metadata.policy_version == "1.0.0"
    assert prediction.has_policy_metadata is True
def test_recovery_decision_exposes_prediction_metadata():
    metadata = RecoveryPolicyMetadata(
        policy_name="deterministic_baseline",
        policy_version="1.0.0",
    )

    prediction = RecoveryPrediction(
        action=RecoveryAction.WAIT,
        confidence=Decimal("0.75"),
        expected_value=Decimal("10.00"),
        reason="Waiting is currently preferred.",
        policy_metadata=metadata,
    )

    outcome = RecoveryOutcome(
        action=RecoveryAction.WAIT,
        payment_status=PaymentStatus.FAILED,
        recovered=False,
        recovered_amount=Decimal("0"),
        recovery_cost=Decimal("0"),
        customer_contacted=False,
        terminal=False,
    )

    decision = RecoveryDecision(
        prediction=prediction,
        outcome=outcome,
    )

    assert decision.action is RecoveryAction.WAIT
    assert decision.confidence == Decimal("0.75")
    assert decision.expected_value == Decimal("10.00")
    assert decision.reason == "Waiting is currently preferred."
    assert decision.policy_metadata == metadata


def test_backward_compatible_recovery_decision_has_no_prediction_metadata():
    outcome = RecoveryOutcome(
        action=RecoveryAction.WAIT,
        payment_status=PaymentStatus.FAILED,
        recovered=False,
        recovered_amount=Decimal("0"),
        recovery_cost=Decimal("0"),
        customer_contacted=False,
        terminal=False,
    )

    decision = RecoveryDecision(
        action=RecoveryAction.WAIT,
        outcome=outcome,
    )

    assert decision.action is RecoveryAction.WAIT
    assert decision.confidence is None
    assert decision.expected_value is None
    assert decision.reason is None
    assert decision.policy_metadata is None

def test_recovery_evaluation_exposes_total_economic_cost():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        available_actions=(
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        ),
    )

    outcome = RecoveryOutcome(
        action=RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        payment_status=PaymentStatus.FAILED,
        recovered=False,
        recovered_amount=Decimal("0"),
        recovery_cost=Decimal("5.00"),
        customer_contacted=True,
        terminal=False,
    )

    decision = RecoveryDecision(
        action=RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        outcome=outcome,
    )

    episode = RecoveryEpisodeResult(
        initial_state=state,
        final_state=state,
        decisions=(decision,),
        recovered=False,
        recovered_amount=Decimal("0"),
        total_recovery_cost=Decimal("5.00"),
        terminal=False,
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        retry_success_probability=Decimal("0.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("5.00"),
        recovery_message_cost=Decimal("2.00"),
        customer_contact_cost=Decimal("3.00"),
        maximum_recovery_attempts=3,
    )

    evaluation = RecoveryEvaluator().evaluate(
        episode=episode,
        scenario=scenario,
    )

    assert evaluation.total_recovery_cost == Decimal("5.00")
    assert evaluation.customer_contact_cost == Decimal("3.00")
    assert evaluation.total_economic_cost == Decimal("8.00")
    assert evaluation.net_recovery_value == Decimal("-8.00")


def test_recovery_evaluation_total_economic_cost_matches_net_value():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.SUCCEEDED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=None,
        failure_code=None,
        available_actions=(),
    )

    outcome = RecoveryOutcome(
        action=RecoveryAction.RETRY_PAYMENT,
        payment_status=PaymentStatus.SUCCEEDED,
        recovered=True,
        recovered_amount=Decimal("1000.00"),
        recovery_cost=Decimal("10.00"),
        customer_contacted=False,
        terminal=True,
    )

    decision = RecoveryDecision(
        action=RecoveryAction.RETRY_PAYMENT,
        outcome=outcome,
    )

    episode = RecoveryEpisodeResult(
        initial_state=state,
        final_state=state,
        decisions=(decision,),
        recovered=True,
        recovered_amount=Decimal("1000.00"),
        total_recovery_cost=Decimal("10.00"),
        terminal=True,
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("10.00"),
        payment_method_update_cost=Decimal("20.00"),
        recovery_message_cost=Decimal("2.00"),
        customer_contact_cost=Decimal("3.00"),
        maximum_recovery_attempts=3,
    )

    evaluation = RecoveryEvaluator().evaluate(
        episode=episode,
        scenario=scenario,
    )

    assert evaluation.total_economic_cost == Decimal("10.00")
    assert evaluation.net_recovery_value == Decimal("990.00")
def test_recovery_policy_predict_returns_structured_prediction():
    class TestPolicy(RecoveryPolicy):
        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            return state.available_actions[0]

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    prediction = TestPolicy().predict(
        state=state,
    )

    assert isinstance(prediction, RecoveryPrediction)
    assert prediction.action is RecoveryAction.WAIT


def test_recovery_policy_rejects_unavailable_predicted_action():
    class TestPolicy(RecoveryPolicy):
        def choose_action(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryAction:
            return RecoveryAction.RETRY_PAYMENT

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    try:
        TestPolicy().predict(
            state=state,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Predicted recovery action 'retry_payment' "
            "is not available for the current recovery state."
        )
    else:
        raise AssertionError(
            "Expected ValueError for unavailable predicted action."
        )
def test_recovery_feature_encoder_feature_names_match_encoded_values():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("750.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("750.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.PAYMENT_METHOD,
        failure_code="card_declined",
        recovery_attempt_count=2,
        previous_actions=(
            RecoveryAction.SEND_RECOVERY_MESSAGE,
        ),
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    encoder = RecoveryFeatureEncoder()

    encoded = encoder.encode(
        features=features,
    )

    names = encoder.feature_names()

    assert len(names) == len(encoded.values)
    assert names == RecoveryFeatureEncoder.FEATURE_NAMES


def test_recovery_feature_encoder_preserves_decimal_feature_values():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1250.50"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1250.50"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        payment_status=payment.status,
        subscription_status=subscription.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        recovery_attempt_count=3,
        previous_actions=(
            RecoveryAction.WAIT,
            RecoveryAction.RETRY_PAYMENT,
        ),
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    features = RecoveryFeatureVector.from_state(
        state=state,
    )

    encoded = RecoveryFeatureEncoder().encode(
        features=features,
    )

    assert encoded.values[0] == Decimal("1250.50")
    assert encoded.values[1] == Decimal("1")
    assert encoded.values[2] == Decimal("0")
    assert encoded.values[3] == Decimal("1")
    assert encoded.values[4] == Decimal("1")
    assert encoded.values[5] == Decimal("0")
    assert encoded.values[6] == Decimal("0")
    assert encoded.values[7] == Decimal("0")
    assert encoded.values[8] == Decimal("3")
    assert encoded.values[9] == Decimal("2")
    assert encoded.values[10] == Decimal("1")
    assert encoded.values[11] == Decimal("0")
    assert encoded.values[12] == Decimal("0")
    assert encoded.values[13] == Decimal("1")
    assert encoded.values[14] == Decimal("0")
def test_model_recovery_policy_passes_encoded_features_to_provider():
    class TestProvider(RecoveryModelProvider):
        def __init__(self) -> None:
            self.received_features = None
            self.received_feature_names = None
            self.received_available_actions = None

        def predict(
            self,
            *,
            features,
            feature_names,
            available_actions,
        ):
            self.received_features = features
            self.received_feature_names = feature_names
            self.received_available_actions = available_actions

            return RecoveryPrediction(
                action=RecoveryAction.WAIT,
                confidence=Decimal("0.80"),
                expected_value=Decimal("0"),
                reason="Provider selected waiting.",
            )

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("750.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("750.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        recovery_attempt_count=2,
        previous_actions=(
            RecoveryAction.SEND_RECOVERY_MESSAGE,
        ),
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    provider = TestProvider()

    policy = ModelRecoveryPolicy(
        provider=provider,
    )

    prediction = policy.predict(
        state=state,
    )

    assert prediction.action is RecoveryAction.WAIT
    assert prediction.confidence == Decimal("0.80")
    assert prediction.expected_value == Decimal("0")
    assert prediction.reason == "Provider selected waiting."

    assert provider.received_features is not None
    assert provider.received_feature_names == (
        RecoveryFeatureEncoder.FEATURE_NAMES
    )
    assert provider.received_available_actions == (
        RecoveryAction.WAIT,
    )

    assert provider.received_features[0] == Decimal("750.00")
    assert provider.received_features[1] == Decimal("1")
    assert provider.received_features[2] == Decimal("0")
    assert provider.received_features[8] == Decimal("2")
    assert provider.received_features[9] == Decimal("1")

    assert prediction.policy_metadata is not None
    assert prediction.policy_metadata.policy_name == (
        "model_recovery_policy"
    )
    assert prediction.policy_metadata.policy_version == "1.0.0"


def test_model_recovery_policy_preserves_provider_policy_metadata():
    provider_metadata = RecoveryPolicyMetadata(
        policy_name="external_model",
        policy_version="2026.08",
    )

    class TestProvider(RecoveryModelProvider):
        def predict(
            self,
            *,
            features,
            feature_names,
            available_actions,
        ):
            return RecoveryPrediction(
                action=RecoveryAction.WAIT,
                confidence=Decimal("0.95"),
                expected_value=Decimal("1.50"),
                reason="External model selected waiting.",
                policy_metadata=provider_metadata,
            )

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    prediction = ModelRecoveryPolicy(
        provider=TestProvider(),
    ).predict(
        state=state,
    )

    assert prediction.policy_metadata == provider_metadata


def test_model_recovery_policy_rejects_provider_action_not_available():
    class TestProvider(RecoveryModelProvider):
        def predict(
            self,
            *,
            features,
            feature_names,
            available_actions,
        ):
            return RecoveryPrediction(
                action=RecoveryAction.RETRY_PAYMENT,
                confidence=Decimal("0.90"),
            )

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("100.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("100.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    try:
        ModelRecoveryPolicy(
            provider=TestProvider(),
        ).predict(
            state=state,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Predicted recovery action 'retry_payment' "
            "is not available for the current recovery state."
        )
    else:
        raise AssertionError(
            "Expected ValueError for unavailable provider action."
        )


def test_ai_provider_config_reads_environment(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setenv("AI_MODEL", "llama3")
    monkeypatch.setenv(
        "AI_BASE_URL",
        "http://localhost:11434",
    )

    config = AIProviderConfig.from_environment()

    assert config.provider == "ollama"
    assert config.api_key == "test-key"
    assert config.model == "llama3"
    assert config.base_url == "http://localhost:11434"


def test_ai_provider_config_allows_missing_api_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setenv("AI_MODEL", "llama3")

    config = AIProviderConfig.from_environment()

    assert config.provider == "ollama"
    assert config.api_key is None
    assert config.model == "llama3"


def test_ai_provider_config_rejects_missing_provider(monkeypatch):
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.setenv("AI_MODEL", "llama3")

    try:
        AIProviderConfig.from_environment()
    except ValueError as exc:
        assert str(exc) == "AI_PROVIDER must be configured."
    else:
        raise AssertionError(
            "Expected ValueError for missing AI provider."
        )


def test_ai_provider_config_rejects_missing_model(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("AI_MODEL", raising=False)

    try:
        AIProviderConfig.from_environment()
    except ValueError as exc:
        assert str(exc) == "AI_MODEL must be configured."
    else:
        raise AssertionError(
            "Expected ValueError for missing AI model."
        )



def test_deterministic_model_provider_prefers_recovery_action():
    provider = DeterministicModelProvider()

    prediction = provider.predict(
        features=(
            Decimal("100.00"),
            Decimal("1"),
        ),
        feature_names=(
            "amount",
            "payment_failed",
        ),
        available_actions=(
            RecoveryAction.WAIT,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            RecoveryAction.RETRY_PAYMENT,
        ),
    )

    assert prediction.action is RecoveryAction.RETRY_PAYMENT
    assert prediction.confidence == Decimal("1.0")
    assert prediction.policy_metadata == provider.metadata
    assert prediction.reason == (
        "Deterministic local model provider selected "
        "'retry_payment'."
    )


def test_deterministic_model_provider_falls_back_to_wait():
    provider = DeterministicModelProvider()

    prediction = provider.predict(
        features=(),
        feature_names=(),
        available_actions=(
            RecoveryAction.WAIT,
        ),
    )

    assert prediction.action is RecoveryAction.WAIT
    assert prediction.confidence == Decimal("1.0")
    assert prediction.policy_metadata == provider.metadata


def test_deterministic_model_provider_falls_back_to_stop():
    provider = DeterministicModelProvider()

    prediction = provider.predict(
        features=(),
        feature_names=(),
        available_actions=(
            RecoveryAction.STOP_RECOVERY,
        ),
    )

    assert prediction.action is RecoveryAction.STOP_RECOVERY
    assert prediction.confidence == Decimal("1.0")
    assert prediction.policy_metadata == provider.metadata


def test_deterministic_model_provider_rejects_empty_action_space():
    provider = DeterministicModelProvider()

    try:
        provider.predict(
            features=(),
            feature_names=(),
            available_actions=(),
        )
    except ValueError as exc:
        assert str(exc) == (
            "No viable recovery action is available."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty action space."
        )

def test_model_provider_factory_creates_deterministic_provider():
    config = AIProviderConfig(
        provider="deterministic",
        api_key=None,
        model="local",
    )

    provider = RecoveryModelProviderFactory.create(
        config=config,
    )

    assert isinstance(
        provider,
        DeterministicModelProvider,
    )


def test_model_provider_factory_rejects_unsupported_provider():
    config = AIProviderConfig(
        provider="unsupported",
        api_key=None,
        model="test-model",
    )

    try:
        RecoveryModelProviderFactory.create(
            config=config,
        )
    except ValueError as exc:
        assert str(exc) == (
            "Unsupported AI provider: unsupported"
        )
    else:
        raise AssertionError(
            "Expected ValueError for unsupported AI provider."
        )

def test_recovery_engine_factory_builds_configured_engine():
    config = AIProviderConfig(
        provider="deterministic",
        api_key=None,
        model="local",
    )

    environment = RecoveryEnvironment(
        scenario=RecoveryScenario(
            failure_category=PaymentFailureCategory.TRANSIENT,
            retry_success_probability=Decimal("0.50"),
            payment_method_update_success_probability=Decimal("0.40"),
            retry_cost=Decimal("2.00"),
            payment_method_update_cost=Decimal("3.00"),
            recovery_message_cost=Decimal("1.00"),
            customer_contact_cost=Decimal("1.00"),
            maximum_recovery_attempts=3,
        ),
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngineFactory.create(
        config=config,
        environment=environment,
    )

    assert isinstance(engine, RecoveryEngine)

def test_recovery_engine_factory_runs_configured_model_policy():
    config = AIProviderConfig(
        provider="deterministic",
        api_key=None,
        model="local",
    )

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("1000.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("1000.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.WAIT,
        ),
    )

    environment = RecoveryEnvironment(
        scenario=RecoveryScenario(
            failure_category=PaymentFailureCategory.TRANSIENT,
            retry_success_probability=Decimal("1.0"),
            payment_method_update_success_probability=Decimal("0.0"),
            retry_cost=Decimal("1.00"),
            payment_method_update_cost=Decimal("2.00"),
            recovery_message_cost=Decimal("0.50"),
            customer_contact_cost=Decimal("0.25"),
            maximum_recovery_attempts=3,
        ),
        simulator=RecoverySimulator(seed=42),
    )

    engine = RecoveryEngineFactory.create(
        config=config,
        environment=environment,
    )

    decision = engine.run_once(
        state=state,
    )

    assert decision.action is RecoveryAction.RETRY_PAYMENT
    assert decision.prediction.action is RecoveryAction.RETRY_PAYMENT
    assert decision.prediction.confidence == Decimal("1.0")
    assert decision.prediction.policy_metadata is not None
    assert (
        decision.prediction.policy_metadata.policy_name
        == "deterministic"
    )
    assert (
        decision.prediction.policy_metadata.policy_version
        == "1.0.0"
    )

    assert decision.outcome.action is RecoveryAction.RETRY_PAYMENT
    assert decision.outcome.recovered is True
    assert decision.outcome.payment_status is PaymentStatus.SUCCEEDED
    assert decision.outcome.recovered_amount == Decimal("1000.00")
def test_recovery_prediction_parser_parses_structured_prediction():
    prediction = RecoveryPredictionParser.parse(
        payload={
            "action": "retry_payment",
            "confidence": "0.85",
            "expected_value": "125.50",
            "reason": "Retry has positive expected recovery value.",
            "policy_metadata": {
                "policy_name": "external_model",
                "policy_version": "2026.08",
            },
        }
    )

    assert prediction.action is RecoveryAction.RETRY_PAYMENT
    assert prediction.confidence == Decimal("0.85")
    assert prediction.expected_value == Decimal("125.50")
    assert prediction.reason == (
        "Retry has positive expected recovery value."
    )
    assert prediction.policy_metadata == RecoveryPolicyMetadata(
        policy_name="external_model",
        policy_version="2026.08",
    )


def test_recovery_prediction_parser_allows_optional_fields():
    prediction = RecoveryPredictionParser.parse(
        payload={
            "action": "wait",
        }
    )

    assert prediction.action is RecoveryAction.WAIT
    assert prediction.confidence is None
    assert prediction.expected_value is None
    assert prediction.reason is None
    assert prediction.policy_metadata is None


def test_recovery_prediction_parser_rejects_missing_action():
    try:
        RecoveryPredictionParser.parse(
            payload={}
        )
    except ValueError as exc:
        assert str(exc) == (
            "Provider response must contain an action."
        )
    else:
        raise AssertionError(
            "Expected ValueError for missing action."
        )


def test_recovery_prediction_parser_rejects_unknown_action():
    try:
        RecoveryPredictionParser.parse(
            payload={
                "action": "charge_customer",
            }
        )
    except ValueError as exc:
        assert str(exc) == (
            "Unsupported recovery action: charge_customer"
        )
    else:
        raise AssertionError(
            "Expected ValueError for unsupported action."
        )


def test_recovery_prediction_parser_rejects_non_numeric_confidence():
    try:
        RecoveryPredictionParser.parse(
            payload={
                "action": "wait",
                "confidence": "not-a-number",
            }
        )
    except ValueError as exc:
        assert str(exc) == (
            "Provider response confidence must be numeric."
        )
    else:
        raise AssertionError(
            "Expected ValueError for invalid confidence."
        )
def test_ai_provider_config_allows_missing_base_url(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_MODEL", "llama3")
    monkeypatch.delenv("AI_BASE_URL", raising=False)

    config = AIProviderConfig.from_environment()

    assert config.base_url is None
def test_ai_provider_config_allows_missing_base_url(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_MODEL", "llama3")
    monkeypatch.delenv("AI_BASE_URL", raising=False)

    config = AIProviderConfig.from_environment()

    assert config.base_url is None
def test_batch_recovery_evaluation_aggregates_metrics():
    evaluations = (
        RecoveryEvaluation(
            recovered=True,
            recovered_amount=Decimal("1000.00"),
            total_recovery_cost=Decimal("10.00"),
            customer_contact_cost=Decimal("5.00"),
            total_economic_cost=Decimal("15.00"),
            net_recovery_value=Decimal("985.00"),
            decision_count=2,
            customer_contact_count=1,
            terminal=True,
            policy_metadata=None,
        ),
        RecoveryEvaluation(
            recovered=False,
            recovered_amount=Decimal("0.00"),
            total_recovery_cost=Decimal("20.00"),
            customer_contact_cost=Decimal("5.00"),
            total_economic_cost=Decimal("25.00"),
            net_recovery_value=Decimal("-25.00"),
            decision_count=3,
            customer_contact_count=1,
            terminal=True,
            policy_metadata=None,
        ),
    )

    evaluation = BatchRecoveryEvaluator().evaluate(
        evaluations=evaluations,
    )

    assert isinstance(evaluation, BatchRecoveryEvaluation)
    assert evaluation.evaluation_count == 2
    assert evaluation.recovered_count == 1
    assert evaluation.recovered_amount == Decimal("1000.00")
    assert evaluation.total_recovery_cost == Decimal("30.00")
    assert evaluation.customer_contact_cost == Decimal("10.00")
    assert evaluation.total_economic_cost == Decimal("40.00")
    assert evaluation.net_recovery_value == Decimal("960.00")
    assert evaluation.recovery_rate == Decimal("0.5")
def test_batch_recovery_evaluation_handles_empty_batch():
    evaluation = BatchRecoveryEvaluator().evaluate(
        evaluations=(),
    )

    assert evaluation.evaluation_count == 0
    assert evaluation.recovered_count == 0
    assert evaluation.recovered_amount == Decimal("0")
    assert evaluation.total_recovery_cost == Decimal("0")
    assert evaluation.customer_contact_cost == Decimal("0")
    assert evaluation.total_economic_cost == Decimal("0")
    assert evaluation.net_recovery_value == Decimal("0")
    assert evaluation.recovery_rate == Decimal("0")
def test_batch_recovery_runner_runs_episode_for_each_state():
    class RecordingEpisode:
        def __init__(self):
            self.states: list[RecoveryState] = []

        def run(
            self,
            *,
            initial_state: RecoveryState,
        ) -> RecoveryEpisodeResult:
            self.states.append(initial_state)

            return RecoveryEpisodeResult(
                initial_state=initial_state,
                final_state=initial_state,
                decisions=(),
                recovered=False,
                recovered_amount=Decimal("0"),
                total_recovery_cost=Decimal("0"),
                terminal=False,
            )

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
        ),
    )

    episode = RecordingEpisode()

    runner = BatchRecoveryRunner(
        episode=episode,
    )

    result = runner.run(
        initial_states=(state, state),
    )

    assert isinstance(result, BatchRecoveryResult)
    assert result.episode_count == 2
    assert len(result.episodes) == 2
    assert episode.states == [state, state]
def test_batch_recovery_workflow_processes_each_input():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    state = RecoveryState.create(
        customer_id=customer.id,
        subscription_id=subscription.id,
        payment_id=payment.id,
        subscription_status=subscription.status,
        payment_status=payment.status,
        amount=payment.amount,
        currency=payment.currency,
        payment_attempted_at=payment.attempted_at,
        failure_category=PaymentFailureCategory.TRANSIENT,
        failure_code="timeout",
        available_actions=(
            RecoveryAction.RETRY_PAYMENT,
        ),
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("5.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("2.00"),
        maximum_recovery_attempts=3,
    )

    class SuccessfulEpisode:
        def run(
            self,
            *,
            initial_state: RecoveryState,
        ) -> RecoveryEpisodeResult:
            outcome = RecoveryOutcome(
                action=RecoveryAction.RETRY_PAYMENT,
                payment_status=PaymentStatus.SUCCEEDED,
                recovered=True,
                recovered_amount=initial_state.amount,
                recovery_cost=Decimal("5.00"),
                customer_contacted=False,
                terminal=True,
            )

            decision = RecoveryDecision(
                action=RecoveryAction.RETRY_PAYMENT,
                outcome=outcome,
            )

            return RecoveryEpisodeResult(
                initial_state=initial_state,
                final_state=initial_state,
                decisions=(decision,),
                recovered=True,
                recovered_amount=initial_state.amount,
                total_recovery_cost=Decimal("5.00"),
                terminal=True,
            )

    workflow = BatchRecoveryWorkflow(
        episode=SuccessfulEpisode(),
        evaluator=RecoveryEvaluator(),
        batch_evaluator=BatchRecoveryEvaluator(),
    )

    result = workflow.run(
        inputs=(
            BatchRecoveryInput(
                state=state,
                scenario=scenario,
            ),
            BatchRecoveryInput(
                state=state,
                scenario=scenario,
            ),
        ),
    )

    assert isinstance(result, BatchRecoveryWorkflowResult)
    assert len(result.episode_results) == 2
    assert len(result.evaluations) == 2

    assert isinstance(
        result.batch_evaluation,
        BatchRecoveryEvaluation,
    )

    assert result.batch_evaluation.evaluation_count == 2
    assert result.batch_evaluation.recovered_count == 2
    assert result.batch_evaluation.recovered_amount == Decimal("1000.00")
    assert result.batch_evaluation.net_recovery_value == Decimal("990.00")
def test_recovery_state_factory_builds_state_from_domain_objects():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=PaymentFailureCategory.TRANSIENT,
        code="timeout",
    )

    state = RecoveryStateFactory.create(
        customer=customer,
        subscription=subscription,
        payment=payment,
        failure=failure,
    )

    assert state.customer_id == customer.id
    assert state.subscription_id == subscription.id
    assert state.payment_id == payment.id

    assert state.subscription_status is subscription.status
    assert state.payment_status is payment.status

    assert state.amount == payment.amount
    assert state.currency == payment.currency
    assert state.payment_attempted_at == payment.attempted_at

    assert (
        state.failure_category
        is PaymentFailureCategory.TRANSIENT
    )
    assert state.failure_code == "timeout"

    assert state.recovery_attempt_count == 0
    assert state.previous_actions == ()

    assert state.available_actions == (
        RecoveryAction.RETRY_PAYMENT,
        RecoveryAction.WAIT,
        RecoveryAction.STOP_RECOVERY,
    )
def test_recovery_workflow_executes_complete_recovery_process():
    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=PaymentFailureCategory.TRANSIENT,
        code="timeout",
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("0.0"),
        retry_cost=Decimal("5.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("2.00"),
        maximum_recovery_attempts=3,
    )

    simulator = RecoverySimulator(
        seed=42,
    )

    environment = RecoveryEnvironment(
        scenario=scenario,
        simulator=simulator,
    )

    class RetryPolicy:
        def predict(
            self,
            *,
            state: RecoveryState,
        ) -> RecoveryPrediction:
            return RecoveryPrediction(
                action=RecoveryAction.RETRY_PAYMENT,
            )

    engine = RecoveryEngine(
        policy=RetryPolicy(),
        environment=environment,
    )

    episode = RecoveryEpisode(
        engine=engine,
        maximum_steps=3,
    )

    workflow = RecoveryWorkflow(
        episode=episode,
        evaluator=RecoveryEvaluator(),
    )

    result = workflow.run(
        customer=customer,
        subscription=subscription,
        payment=payment,
        failure=failure,
        scenario=scenario,
    )

    assert isinstance(result, RecoveryWorkflowResult)
    assert result.episode.recovered is True
    assert result.episode.recovered_amount == Decimal("500.00")
    assert result.evaluation.recovered is True
    assert result.evaluation.net_recovery_value == Decimal("495.00")
def test_recovery_workflow_factory_creates_workflow():
    config = AIProviderConfig(
        provider="deterministic",
        api_key=None,
        model="local",
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("1.0"),
        retry_cost=Decimal("5.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("2.00"),
        maximum_recovery_attempts=3,
    )

    workflow = RecoveryWorkflowFactory.create(
        config=config,
        scenario=scenario,
        simulator_seed=42,
        maximum_steps=3,
    )

    assert isinstance(workflow, RecoveryWorkflow)
def test_recovery_workflow_factory_runs_complete_recovery_workflow():
    config = AIProviderConfig(
        provider="deterministic",
        api_key=None,
        model="local",
    )

    scenario = RecoveryScenario(
        failure_category=PaymentFailureCategory.TRANSIENT,
        retry_success_probability=Decimal("1.0"),
        payment_method_update_success_probability=Decimal("1.0"),
        retry_cost=Decimal("5.00"),
        payment_method_update_cost=Decimal("10.00"),
        recovery_message_cost=Decimal("1.00"),
        customer_contact_cost=Decimal("2.00"),
        maximum_recovery_attempts=3,
    )

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=Decimal("500.00"),
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=Decimal("500.00"),
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=PaymentFailureCategory.TRANSIENT,
        code="temporary_failure",
    )

    workflow = RecoveryWorkflowFactory.create(
        config=config,
        scenario=scenario,
        simulator_seed=42,
        maximum_steps=3,
    )

    result = workflow.run(
        customer=customer,
        subscription=subscription,
        payment=payment,
        failure=failure,
        scenario=scenario,
    )

    assert isinstance(result, RecoveryWorkflowResult)
    assert result.episode.recovered is True
    assert result.episode.recovered_amount == Decimal("500.00")
    assert result.evaluation.recovered is True
    assert result.evaluation.net_recovery_value == Decimal("495.00")
    
def test_cli_main_runs_complete_recovery(
    monkeypatch,
    capsys,
):
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    main()

    captured = capsys.readouterr()

    assert "RecoverIQ Recovery Result" in captured.out
    assert "Recovered: True" in captured.out
    assert "Recovered amount: 500.00" in captured.out
    assert "Total recovery cost: 5.00" in captured.out
    assert "Net recovery value: 495.00" in captured.out
    assert "Decision count: 1" in captured.out
    assert "Terminal: True" in captured.out
    
def test_cli_main_runs_with_custom_amount(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recoveriq.cli",
            "--amount",
            "1000",
        ],
    )

    from recoveriq.cli import main

    main()

    captured = capsys.readouterr()

    assert "Payment amount: 1000" in captured.out
    assert "Recovered amount: 1000" in captured.out
    assert "Net recovery value: 995.00" in captured.out

def test_cli_main_runs_with_custom_maximum_steps(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recoveriq.cli",
            "--maximum-steps",
            "5",
        ],
    )

    from recoveriq.cli import main

    main()

    captured = capsys.readouterr()

    assert "Maximum steps: 5" in captured.out
    assert "Recovered: True" in captured.out
    assert "Terminal: True" in captured.out
    
def test_cli_main_runs_with_custom_simulator_seed(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recoveriq.cli",
            "--simulator-seed",
            "99",
        ],
    )

    from recoveriq.cli import main

    main()

    captured = capsys.readouterr()

    assert "Simulator seed: 99" in captured.out
    assert "Recovered: True" in captured.out
    assert "Terminal: True" in captured.out
    
def test_cli_main_runs_with_custom_retry_success_probability(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recoveriq.cli",
            "--retry-success-probability",
            "0.5",
            "--simulator-seed",
            "42",
        ],
    )

    from recoveriq.cli import main

    main()

    captured = capsys.readouterr()

    assert "Retry success probability: 0.5" in captured.out
    assert "Simulator seed: 42" in captured.out
    assert "Recovered: False" in captured.out
    assert "Terminal: True" in captured.out