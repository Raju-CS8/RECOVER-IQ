from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata


@dataclass(frozen=True, slots=True)
class RecoveryPrediction:
    """
    Immutable prediction produced by a recovery decision policy.

    A prediction contains the selected action together with optional
    decision metadata and policy provenance.
    """

    action: RecoveryAction
    confidence: Decimal | None = None
    expected_value: Decimal | None = None
    reason: str | None = None
    policy_metadata: RecoveryPolicyMetadata | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None:
            if not Decimal("0") <= self.confidence <= Decimal("1"):
                raise ValueError(
                    "Prediction confidence must be between zero and one."
                )

        if self.reason is not None and not self.reason.strip():
            raise ValueError(
                "Prediction reason must not be empty."
            )

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None

    @property
    def has_expected_value(self) -> bool:
        return self.expected_value is not None

    @property
    def has_reason(self) -> bool:
        return self.reason is not None

    @property
    def has_policy_metadata(self) -> bool:
        return self.policy_metadata is not None