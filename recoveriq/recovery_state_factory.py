from __future__ import annotations

from recoveriq.domain.customer import Customer
from recoveriq.domain.payment import Payment
from recoveriq.domain.payment_failure import (
    PaymentFailure,
    PaymentFailureCategory,
)
from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_state import RecoveryState
from recoveriq.domain.subscription import Subscription


class RecoveryStateFactory:
    """
    Builds recovery decision states from existing domain objects.

    The factory converts customer, subscription, payment, and failure
    information into the limited decision-state representation required
    by the recovery engine.
    """

    @classmethod
    def create(
        cls,
        *,
        customer: Customer,
        subscription: Subscription,
        payment: Payment,
        failure: PaymentFailure,
    ) -> RecoveryState:
        """
        Build a recovery state from related domain objects.
        """

        cls._validate_relationships(
            customer=customer,
            subscription=subscription,
            payment=payment,
            failure=failure,
        )

        return RecoveryState.create(
            customer_id=customer.id,
            subscription_id=subscription.id,
            payment_id=payment.id,
            subscription_status=subscription.status,
            payment_status=payment.status,
            amount=payment.amount,
            currency=payment.currency,
            payment_attempted_at=payment.attempted_at,
            failure_category=failure.category,
            failure_code=failure.code,
            available_actions=cls._available_actions(
                failure_category=failure.category,
            ),
        )

    @staticmethod
    def _validate_relationships(
        *,
        customer: Customer,
        subscription: Subscription,
        payment: Payment,
        failure: PaymentFailure,
    ) -> None:
        """
        Ensure the supplied domain objects belong to the same payment flow.
        """

        if subscription.customer_id != customer.id:
            raise ValueError(
                "Subscription does not belong to the supplied customer."
            )

        if payment.subscription_id != subscription.id:
            raise ValueError(
                "Payment does not belong to the supplied subscription."
            )

        if failure.payment_id != payment.id:
            raise ValueError(
                "Payment failure does not belong to the supplied payment."
            )

    @staticmethod
    def _available_actions(
        *,
        failure_category: PaymentFailureCategory,
    ) -> tuple[RecoveryAction, ...]:
        """
        Determine the permitted recovery actions for a failure category.
        """

        if failure_category is PaymentFailureCategory.TRANSIENT:
            return (
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.WAIT,
                RecoveryAction.STOP_RECOVERY,
            )

        if failure_category is PaymentFailureCategory.PAYMENT_METHOD:
            return (
                RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
                RecoveryAction.SEND_RECOVERY_MESSAGE,
                RecoveryAction.STOP_RECOVERY,
            )

        if failure_category is PaymentFailureCategory.AUTHENTICATION:
            return (
                RecoveryAction.SEND_RECOVERY_MESSAGE,
                RecoveryAction.WAIT,
                RecoveryAction.STOP_RECOVERY,
            )

        return (
            RecoveryAction.SEND_RECOVERY_MESSAGE,
            RecoveryAction.WAIT,
            RecoveryAction.STOP_RECOVERY,
        )