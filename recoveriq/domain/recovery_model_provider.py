from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_prediction import RecoveryPrediction


class RecoveryModelProvider(ABC):
    """
    Contract for a model or decision provider used by an AI recovery policy.

    Implementations may use machine learning, optimization, an external AI
    service, or another decision algorithm.

    The provider receives encoded numeric features and returns a structured
    recovery prediction.

    It does not execute recovery actions or mutate domain state.
    """

    @abstractmethod
    def predict(
        self,
        *,
        features: tuple[Decimal, ...],
        feature_names: tuple[str, ...],
        available_actions: tuple[RecoveryAction, ...],
    ) -> RecoveryPrediction:
        """
        Produce a recovery prediction from encoded model features.
        """

        raise NotImplementedError
