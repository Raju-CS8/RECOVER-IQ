from __future__ import annotations

from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_model_provider import RecoveryModelProvider
from recoveriq.domain.recovery_prediction import RecoveryPrediction
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata


class DeterministicModelProvider(RecoveryModelProvider):
    """
    Local deterministic model provider used for development and testing.

    This provider does not perform network calls or invoke an external AI
    service. It provides a stable model-provider implementation so the
    recovery policy pipeline can be exercised independently of infrastructure.
    """

    PROVIDER_NAME = "deterministic"
    PROVIDER_VERSION = "1.0.0"

    @property
    def metadata(self) -> RecoveryPolicyMetadata:
        """
        Return stable provider identity metadata.
        """

        return RecoveryPolicyMetadata(
            policy_name=self.PROVIDER_NAME,
            policy_version=self.PROVIDER_VERSION,
        )

    def predict(
        self,
        *,
        features: tuple[Decimal, ...],
        feature_names: tuple[str, ...],
        available_actions: tuple[RecoveryAction, ...],
    ) -> RecoveryPrediction:
        """
        Produce a deterministic prediction from encoded features.

        The provider currently prefers recovery-producing actions according
        to a fixed priority and falls back to WAIT or STOP_RECOVERY.
        """

        del features
        del feature_names

        priority = (
            RecoveryAction.RETRY_PAYMENT,
            RecoveryAction.REQUEST_PAYMENT_METHOD_UPDATE,
            RecoveryAction.SEND_RECOVERY_MESSAGE,
            RecoveryAction.WAIT,
            RecoveryAction.STOP_RECOVERY,
        )

        for action in priority:
            if action in available_actions:
                return RecoveryPrediction(
                    action=action,
                    confidence=Decimal("1.0"),
                    expected_value=None,
                    reason=(
                        "Deterministic local model provider selected "
                        f"'{action.value}'."
                    ),
                    policy_metadata=self.metadata,
                )

        raise ValueError(
            "No viable recovery action is available."
        )
