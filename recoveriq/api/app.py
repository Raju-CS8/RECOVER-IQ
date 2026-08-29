from fastapi import FastAPI

from recoveriq.ai_config import AIProviderConfig
from recoveriq.api.schemas import (
    BatchRecoveryItemResponse,
    BatchRecoveryRequest,
    BatchRecoveryResponse,
    RecoveryDecisionResponse,
    RecoveryRequest,
    RecoveryResponse,
)
from recoveriq.application.batch_recovery_workflow_factory import (
    BatchRecoveryWorkflowFactory,
)
from recoveriq.application.recovery_workflow_factory import (
    RecoveryWorkflowFactory,
)
from recoveriq.domain.batch_recovery_workflow import (
    BatchRecoveryInput,
)
from recoveriq.domain.customer import Customer
from recoveriq.domain.payment import Payment, PaymentStatus
from recoveriq.domain.payment_failure import PaymentFailure
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.subscription import Subscription
from recoveriq.recovery_state_factory import RecoveryStateFactory


app = FastAPI(
    title="RecoverIQ API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {
        "status": "ok",
    }


@app.post(
    "/recover",
    response_model=RecoveryResponse,
)
def recover_payment(
    request: RecoveryRequest,
) -> RecoveryResponse:
    """Run one complete payment recovery workflow."""

    config = AIProviderConfig.from_environment()

    customer = Customer.create()

    subscription = Subscription.create(
        customer_id=customer.id,
        amount=request.amount,
        currency="INR",
    )

    payment = Payment.create(
        subscription_id=subscription.id,
        amount=request.amount,
        currency="INR",
        status=PaymentStatus.FAILED,
    )

    failure = PaymentFailure.create(
        payment_id=payment.id,
        category=request.failure_category,
        code=f"{request.failure_category.value}_failure",
    )

    scenario = RecoveryScenario(
        failure_category=request.failure_category,
        retry_success_probability=(
            request.retry_success_probability
        ),
        payment_method_update_success_probability=(
            request.payment_method_update_success_probability
        ),
        retry_cost=request.retry_cost,
        payment_method_update_cost=(
            request.payment_method_update_cost
        ),
        recovery_message_cost=request.recovery_message_cost,
        customer_contact_cost=request.customer_contact_cost,
        maximum_recovery_attempts=(
            request.maximum_recovery_attempts
        ),
    )

    workflow = RecoveryWorkflowFactory.create(
        config=config,
        scenario=scenario,
        simulator_seed=request.simulator_seed,
        maximum_steps=request.maximum_steps,
    )

    result = workflow.run(
        customer=customer,
        subscription=subscription,
        payment=payment,
        failure=failure,
        scenario=scenario,
    )

    evaluation = result.evaluation

    decisions = [
        RecoveryDecisionResponse(
            action=decision.action.value,
            expected_value=decision.expected_value,
            confidence=decision.confidence,
            reason=decision.reason,
            recovered=decision.outcome.recovered,
            recovered_amount=decision.outcome.recovered_amount,
            recovery_cost=decision.outcome.recovery_cost,
            customer_contacted=(
                decision.outcome.customer_contacted
            ),
            terminal=decision.outcome.terminal,
        )
        for decision in result.episode.decisions
    ]

    return RecoveryResponse(
        recovered=evaluation.recovered,
        recovered_amount=evaluation.recovered_amount,
        total_recovery_cost=(
            evaluation.total_recovery_cost
        ),
        customer_contact_cost=(
            evaluation.customer_contact_cost
        ),
        total_economic_cost=(
            evaluation.total_economic_cost
        ),
        net_recovery_value=(
            evaluation.net_recovery_value
        ),
        decision_count=evaluation.decision_count,
        customer_contact_count=(
            evaluation.customer_contact_count
        ),
        terminal=evaluation.terminal,
        decisions=decisions,
    )


@app.post(
    "/recover/batch",
    response_model=BatchRecoveryResponse,
)
def recover_payment_batch(
    request: BatchRecoveryRequest,
) -> BatchRecoveryResponse:
    """
    Run recovery workflows for multiple failed payments.

    Each batch item receives its own recovery state and scenario while
    sharing the configured batch-level recovery parameters.
    """

    config = AIProviderConfig.from_environment()

    inputs: list[BatchRecoveryInput] = []

    for item in request.items:
        customer = Customer.create()

        subscription = Subscription.create(
            customer_id=customer.id,
            amount=item.amount,
            currency="INR",
        )

        payment = Payment.create(
            subscription_id=subscription.id,
            amount=item.amount,
            currency="INR",
            status=PaymentStatus.FAILED,
        )

        failure = PaymentFailure.create(
            payment_id=payment.id,
            category=item.failure_category,
            code=(
                f"{item.failure_category.value}_failure"
            ),
        )

        state = RecoveryStateFactory.create(
            customer=customer,
            subscription=subscription,
            payment=payment,
            failure=failure,
        )

        scenario = RecoveryScenario(
            failure_category=item.failure_category,
            retry_success_probability=(
                request.retry_success_probability
            ),
            payment_method_update_success_probability=(
                request.payment_method_update_success_probability
            ),
            retry_cost=request.retry_cost,
            payment_method_update_cost=(
                request.payment_method_update_cost
            ),
            recovery_message_cost=(
                request.recovery_message_cost
            ),
            customer_contact_cost=(
                request.customer_contact_cost
            ),
            maximum_recovery_attempts=(
                request.maximum_recovery_attempts
            ),
        )

        inputs.append(
            BatchRecoveryInput(
                state=state,
                scenario=scenario,
            )
        )

    workflow = BatchRecoveryWorkflowFactory.create(
        config=config,
        simulator_seed=request.simulator_seed,
        maximum_steps=request.maximum_steps,
    )

    result = workflow.run(
        inputs=tuple(inputs),
    )

    items = [
        BatchRecoveryItemResponse(
            recovered=evaluation.recovered,
            recovered_amount=(
                evaluation.recovered_amount
            ),
            total_recovery_cost=(
                evaluation.total_recovery_cost
            ),
            customer_contact_cost=(
                evaluation.customer_contact_cost
            ),
            total_economic_cost=(
                evaluation.total_economic_cost
            ),
            net_recovery_value=(
                evaluation.net_recovery_value
            ),
            decision_count=(
                evaluation.decision_count
            ),
            customer_contact_count=(
                evaluation.customer_contact_count
            ),
            terminal=evaluation.terminal,
        )
        for evaluation in result.evaluations
    ]

    batch_evaluation = result.batch_evaluation

    return BatchRecoveryResponse(
        evaluation_count=(
            batch_evaluation.evaluation_count
        ),
        recovered_count=(
            batch_evaluation.recovered_count
        ),
        recovered_amount=(
            batch_evaluation.recovered_amount
        ),
        total_recovery_cost=(
            batch_evaluation.total_recovery_cost
        ),
        customer_contact_cost=(
            batch_evaluation.customer_contact_cost
        ),
        total_economic_cost=(
            batch_evaluation.total_economic_cost
        ),
        net_recovery_value=(
            batch_evaluation.net_recovery_value
        ),
        recovery_rate=(
            batch_evaluation.recovery_rate
        ),
        items=items,
    )