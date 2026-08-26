from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.payment_failure import PaymentFailureCategory
from recoveriq.domain.recovery_features import RecoveryFeatureVector


@dataclass(frozen=True, slots=True)
class EncodedRecoveryFeatures:
    """
    Immutable numeric representation of a RecoveryFeatureVector.

    The ordering of fields is part of the model-input contract and must
    remain stable unless the feature representation is intentionally
    versioned.
    """

    values: tuple[Decimal, ...]


class RecoveryFeatureEncoder:
    """
    Deterministically converts RecoveryFeatureVector into numeric features.

    This class performs representation only.

    It does not:
    - train a model
    - choose an action
    - estimate probabilities
    - access external services
    - introduce randomness
    """

    FEATURE_NAMES: tuple[str, ...] = (
        "amount",
        "payment_failed",
        "payment_succeeded",
        "subscription_active",
        "failure_transient",
        "failure_payment_method",
        "failure_authentication",
        "failure_unknown",
        "recovery_attempt_count",
        "previous_action_count",
        "retry_available",
        "payment_method_update_available",
        "recovery_message_available",
        "wait_available",
        "stop_recovery_available",
    )

    def encode(
        self,
        *,
        features: RecoveryFeatureVector,
    ) -> EncodedRecoveryFeatures:
        """
        Encode a feature vector into a stable numeric tuple.
        """

        return EncodedRecoveryFeatures(
            values=(
                features.amount,
                self._boolean(features.payment_failed),
                self._boolean(features.payment_succeeded),
                self._boolean(features.subscription_active),
                self._failure_category(
                    features.failure_category,
                    PaymentFailureCategory.TRANSIENT,
                ),
                self._failure_category(
                    features.failure_category,
                    PaymentFailureCategory.PAYMENT_METHOD,
                ),
                self._failure_category(
                    features.failure_category,
                    PaymentFailureCategory.AUTHENTICATION,
                ),
                self._failure_category(
                    features.failure_category,
                    PaymentFailureCategory.UNKNOWN,
                ),
                Decimal(features.recovery_attempt_count),
                Decimal(features.previous_action_count),
                self._boolean(features.retry_available),
                self._boolean(
                    features.payment_method_update_available
                ),
                self._boolean(
                    features.recovery_message_available
                ),
                self._boolean(features.wait_available),
                self._boolean(features.stop_recovery_available),
            )
        )

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        """
        Return the stable ordering of encoded features.
        """

        return cls.FEATURE_NAMES

    @staticmethod
    def _boolean(value: bool) -> Decimal:
        return Decimal("1") if value else Decimal("0")

    @staticmethod
    def _failure_category(
        actual: PaymentFailureCategory | None,
        expected: PaymentFailureCategory,
    ) -> Decimal:
        if actual is expected:
            return Decimal("1")

        return Decimal("0")