from __future__ import annotations

from abc import ABC, abstractmethod

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_prediction import RecoveryPrediction
from recoveriq.domain.recovery_state import RecoveryState


class RecoveryPolicy(ABC):
    """
    Contract for recovery decision policies.

    A policy may be deterministic, optimization-based, statistical,
    machine-learning-based, or AI-driven.

    The policy selects decisions. It does not execute recovery actions
    or mutate the recovery environment.
    """

    def predict(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryPrediction:
        """
        Produce and validate a structured recovery prediction.
        """

        prediction = self._predict(
            state=state,
        )

        if prediction.action not in state.available_actions:
            raise ValueError(
                f"Predicted recovery action '{prediction.action.value}' "
                "is not available for the current recovery state."
            )

        return prediction

    def _predict(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryPrediction:
        """
        Internal prediction hook.

        The default implementation adapts the existing action-selection
        contract into a RecoveryPrediction.
        """

        action = self.choose_action(
            state=state,
        )

        return RecoveryPrediction(
            action=action,
        )

    @abstractmethod
    def choose_action(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryAction:
        """
        Select one recovery action for the supplied state.
        """

        raise NotImplementedError