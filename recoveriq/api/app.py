from fastapi import FastAPI

from recoveriq.ai_config import AIProviderConfig
from recoveriq.api.schemas import (
    RecoveryDecisionResponse,
    RecoveryRequest,
    RecoveryResponse,
)
from recoveriq.application.recovery_workflow_factory import (
    RecoveryWorkflowFactory,
)
from recoveriq.domain.customer import Customer
from recoveriq.domain.payment import Payment, PaymentStatus
from recoveriq.domain.payment_failure import PaymentFailure
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.subscription import Subscription


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
