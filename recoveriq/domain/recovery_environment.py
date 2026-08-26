from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.payment import PaymentStatus
from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_simulator import RecoverySimulator
from recoveriq.domain.recovery_state import RecoveryState


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """
    Immutable result produced by the recovery environment after executing
    a recovery action.

    The decision engine does not construct this object. The environment
    produces it from the action, scenario, and resulting simulation.
    """

    action: RecoveryAction
    payment_status: PaymentStatus
    recovered: bool
    recovered_amount: Decimal
    recovery_cost: Decimal
    customer_contacted: bool
    terminal: bool


class RecoveryEnvironment:
    """
    Deterministic recovery execution boundary.

    The environment owns:
    - action validation
    - scenario execution
    - simulation
    - outcome calculation

    A decision engine may request an action, but it cannot directly mutate
    payment state or declare the result of that action.
    """

    def __init__(
        self,
        *,
        scenario: RecoveryScenario,
        simulator: RecoverySimulator,
    ) -> None:
        self._scenario = scenario
        self._simulator = simulator

    def execute(
        self,
        *,
        state: RecoveryState,
        action: RecoveryAction,
    ) -> RecoveryOutcome:
        """
        Execute one recovery action against the current recovery state.
        """

        if action not in state.available_actions:
            raise ValueError(
                f"Recovery action '{action.value}' is not available "
                "for the current recovery state."
            )

        if state.payment_status is PaymentStatus.SUCCEEDED:
            raise ValueError(
                "Recovery actions cannot be executed for a succeeded payment."
            )

        if (
            state.recovery_attempt_count
            >= self._scenario.maximum_recovery_attempts
            and action
            in {
                RecoveryAction.RETRY_PAYMENT,
                RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            }
        ):
            raise ValueError(
                "Maximum recovery attempts have been reached."
            )

        if action is RecoveryAction.STOP_RECOVERY:
            return RecoveryOutcome(
                action=action,
                payment_status=state.payment_status,
                recovered=False,
                recovered_amount=Decimal("0"),
                recovery_cost=Decimal("0"),
                customer_contacted=False,
                terminal=True,
            )

        if action is RecoveryAction.WAIT:
            return RecoveryOutcome(
                action=action,
                payment_status=state.payment_status,
                recovered=False,
                recovered_amount=Decimal("0"),
                recovery_cost=Decimal("0"),
                customer_contacted=False,
                terminal=False,
            )

        if action is RecoveryAction.SEND_RECOVERY_MESSAGE:
            return RecoveryOutcome(
                action=action,
                payment_status=state.payment_status,
                recovered=False,
                recovered_amount=Decimal("0"),
                recovery_cost=self._scenario.recovery_message_cost,
                customer_contacted=True,
                terminal=False,
            )

        if action is RecoveryAction.RETRY_PAYMENT:
            simulation = self._simulator.trial(
                success_probability=self._scenario.retry_success_probability,
            )

            if simulation.successful:
                return RecoveryOutcome(
                    action=action,
                    payment_status=PaymentStatus.SUCCEEDED,
                    recovered=True,
                    recovered_amount=state.amount,
                    recovery_cost=self._scenario.retry_cost,
                    customer_contacted=False,
                    terminal=True,
                )

            return RecoveryOutcome(
                action=action,
                payment_status=PaymentStatus.FAILED,
                recovered=False,
                recovered_amount=Decimal("0"),
                recovery_cost=self._scenario.retry_cost,
                customer_contacted=False,
                terminal=False,
            )

        if action is RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE:
            simulation = self._simulator.trial(
                success_probability=(
                    self._scenario.payment_method_update_success_probability
                ),
            )

            if simulation.successful:
                return RecoveryOutcome(
                    action=action,
                    payment_status=PaymentStatus.SUCCEEDED,
                    recovered=True,
                    recovered_amount=state.amount,
                    recovery_cost=self._scenario.payment_method_update_cost,
                    customer_contacted=True,
                    terminal=True,
                )

            return RecoveryOutcome(
                action=action,
                payment_status=PaymentStatus.FAILED,
                recovered=False,
                recovered_amount=Decimal("0"),
                recovery_cost=self._scenario.payment_method_update_cost,
                customer_contacted=True,
                terminal=False,
            )

        raise ValueError(
            f"Unsupported recovery action: {action.value}"
        )