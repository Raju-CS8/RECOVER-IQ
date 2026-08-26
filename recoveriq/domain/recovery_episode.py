from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.payment import PaymentStatus
from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_engine import RecoveryDecision, RecoveryEngine
from recoveriq.domain.recovery_state import RecoveryState


@dataclass(frozen=True)
class RecoveryEpisodeResult:
    """
    Immutable result of a complete recovery episode.

    The episode records every decision made during the recovery process
    and exposes the final recovery state and aggregate recovery metrics.
    """

    initial_state: RecoveryState
    final_state: RecoveryState
    decisions: tuple[RecoveryDecision, ...]
    recovered: bool
    recovered_amount: Decimal
    total_recovery_cost: Decimal
    terminal: bool

    @property
    def decision_count(self) -> int:
        return len(self.decisions)


class RecoveryEpisode:
    """
    Executes a bounded sequence of recovery decisions.

    The episode is responsible for orchestration and lifecycle management.
    It does not decide which action is best and does not simulate payment
    outcomes itself.

    Responsibilities:

        RecoveryDecisionPolicy
            -> chooses action

        RecoveryEnvironment
            -> executes action

        RecoveryEngine
            -> coordinates one decision cycle

        RecoveryEpisode
            -> coordinates multiple decision cycles
    """

    def __init__(
        self,
        *,
        engine: RecoveryEngine,
        maximum_steps: int,
    ) -> None:
        if maximum_steps <= 0:
            raise ValueError("Maximum episode steps must be greater than zero.")

        self._engine = engine
        self._maximum_steps = maximum_steps

    def run(
        self,
        *,
        initial_state: RecoveryState,
    ) -> RecoveryEpisodeResult:
        """
        Execute the recovery process until it reaches a terminal outcome
        or the configured episode step limit.
        """

        decisions: list[RecoveryDecision] = []
        current_state = initial_state
        total_recovery_cost = Decimal("0")

        for _ in range(self._maximum_steps):
            decision = self._engine.run_once(
                state=current_state,
            )

            decisions.append(decision)
            total_recovery_cost += decision.outcome.recovery_cost

            current_state = self._next_state(
                state=current_state,
                decision=decision,
            )

            if decision.outcome.terminal:
                break

        final_outcome = decisions[-1].outcome

        return RecoveryEpisodeResult(
            initial_state=initial_state,
            final_state=current_state,
            decisions=tuple(decisions),
            recovered=final_outcome.recovered,
            recovered_amount=final_outcome.recovered_amount,
            total_recovery_cost=total_recovery_cost,
            terminal=final_outcome.terminal,
        )

    @staticmethod
    def _next_state(
        *,
        state: RecoveryState,
        decision: RecoveryDecision,
    ) -> RecoveryState:
        outcome = decision.outcome

        previous_actions = (
            *state.previous_actions,
            decision.action,
        )

        recovery_attempt_count = state.recovery_attempt_count

        if decision.action in {
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
        }:
            recovery_attempt_count += 1

        if outcome.payment_status is PaymentStatus.SUCCEEDED:
            available_actions: tuple[RecoveryAction, ...] = ()
        elif outcome.terminal:
            available_actions = ()
        else:
            available_actions = tuple(
                action
                for action in state.available_actions
                if action is not decision.action
            )

        return RecoveryState.create(
            customer_id=state.customer_id,
            subscription_id=state.subscription_id,
            payment_id=state.payment_id,
            subscription_status=state.subscription_status,
            payment_status=outcome.payment_status,
            amount=state.amount,
            currency=state.currency,
            payment_attempted_at=state.payment_attempted_at,
            failure_category=state.failure_category,
            failure_code=state.failure_code,
            recovery_attempt_count=recovery_attempt_count,
            previous_actions=previous_actions,
            available_actions=available_actions,
        )