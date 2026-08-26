from __future__ import annotations

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_feature_encoder import RecoveryFeatureEncoder
from recoveriq.domain.recovery_features import RecoveryFeatureVector
from recoveriq.domain.recovery_model_provider import RecoveryModelProvider
from recoveriq.domain.recovery_policy_interface import RecoveryPolicy
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata
from recoveriq.domain.recovery_prediction import RecoveryPrediction
from recoveriq.domain.recovery_state import RecoveryState


class ModelRecoveryPolicy(RecoveryPolicy):
    """
    Recovery policy backed by a model/provider.

    The policy owns the translation between domain recovery state and the
    model-facing feature representation. The provider owns the actual
    prediction logic.

    The provider cannot execute recovery actions or mutate domain state.
    """

    POLICY_NAME = "model_recovery_policy"
    POLICY_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        provider: RecoveryModelProvider,
        encoder: RecoveryFeatureEncoder | None = None,
    ) -> None:
        self._provider = provider
        self._encoder = encoder or RecoveryFeatureEncoder()

    @property
    def metadata(self) -> RecoveryPolicyMetadata:
        """
        Return stable identity information for this policy.
        """

        return RecoveryPolicyMetadata(
            policy_name=self.POLICY_NAME,
            policy_version=self.POLICY_VERSION,
        )

    def choose_action(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryAction:
        """
        Return the action selected by the model provider.
        """

        return self.predict(
            state=state,
        ).action

    def _predict(
        self,
        *,
        state: RecoveryState,
    ) -> RecoveryPrediction:
        """
        Encode the recovery state and delegate prediction to the provider.
        """

        features = RecoveryFeatureVector.from_state(
            state=state,
        )

        encoded = self._encoder.encode(
            features=features,
        )

        prediction = self._provider.predict(
            features=encoded.values,
            feature_names=self._encoder.feature_names(),
            available_actions=state.available_actions,
        )

        if prediction.policy_metadata is not None:
            return prediction

        return RecoveryPrediction(
            action=prediction.action,
            confidence=prediction.confidence,
            expected_value=prediction.expected_value,
            reason=prediction.reason,
            policy_metadata=self.metadata,
        )
