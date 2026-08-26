from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_environment import (
    RecoveryEnvironment,
    RecoveryOutcome,
)
from recoveriq.domain.recovery_policy_interface import RecoveryPolicy
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata
from recoveriq.domain.recovery_prediction import RecoveryPrediction
from recoveriq.domain.recovery_state import RecoveryState


@dataclass(frozen=True, slots=True, init=False)
class RecoveryDecision:
    """
    Immutable record containing the selected recovery action, policy
    prediction, and resulting recovery-environment outcome.

    The action-based constructor remains supported for backward
    compatibility with existing domain consumers.
    """

    prediction: RecoveryPrediction
    outcome: RecoveryOutcome

    def __init__(
        self,
        *,
        outcome: RecoveryOutcome,
        action: RecoveryAction | None = None,
        prediction: RecoveryPrediction | None = None,
    ) -> None:
        if action is not None and prediction is not None:
            if action is not prediction.action:
                raise ValueError(
                    "Decision action must match prediction action."
                )

        if prediction is None:
            if action is None:
                raise ValueError(
                    "RecoveryDecision requires action or prediction."
                )

            prediction = RecoveryPrediction(
                action=action,
            )

        if prediction.action is not outcome.action:
            raise ValueError(
                "Decision action must match outcome action."
            )

        object.__setattr__(
            self,
            "prediction",
            prediction,
        )
        object.__setattr__(
            self,
            "outcome",
            outcome,
        )

    @property
    def action(self) -> RecoveryAction:
        """
        Return the action selected by the policy prediction.
        """

        return self.prediction.action

    @property
    def expected_value(self) -> Decimal | None:
        """
        Return the expected recovery value associated with the prediction.
        """

        return self.prediction.expected_value

    @property
    def confidence(self) -> Decimal | None:
        """
        Return the prediction confidence when supplied by the policy.
        """

        return self.prediction.confidence

    @property
    def reason(self) -> str | None:
        """
        Return the policy's explanation for the selected action.
        """

        return self.prediction.reason

    @property
    def policy_metadata(self) -> RecoveryPolicyMetadata | None:
        """
        Return policy provenance associated with the prediction.
        """

        return self.prediction.policy_metadata


class RecoveryEngine:
    """
    Coordinates recovery decision-making and recovery execution.

    The engine deliberately keeps policy and environment responsibilities
    separate:

        RecoveryPolicy
            -> produces RecoveryPrediction

        RecoveryEnvironment
            -> executes RecoveryAction

        RecoveryEngine
            -> coordinates prediction and execution
    """

    def __init__(
        self,
        *,
        policy: RecoveryPolicy,
        environment: RecoveryEnvironment,
    ) -> None:
        self._policy = policy
        self._environment = environment

    def run_once(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryDecision:
        """
        Execute exactly one recovery decision cycle.
        """

        prediction = self._policy.predict(
            state=state,
        )

        outcome = self._environment.execute(
            state=state,
            action=prediction.action,
        )

        return RecoveryDecision(
            prediction=prediction,
            outcome=outcome,
        )