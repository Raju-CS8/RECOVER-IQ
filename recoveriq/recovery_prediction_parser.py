from __future__ import annotations

from decimal import Decimal
from typing import Any

from recoveriq.domain.recovery_action import RecoveryAction
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata
from recoveriq.domain.recovery_prediction import RecoveryPrediction


class RecoveryPredictionParser:
    """
    Converts structured provider output into a validated
    RecoveryPrediction.

    Provider-specific transport and network concerns remain outside this
    parser. The parser only validates and translates structured data.
    """

    @staticmethod
    def parse(
        *,
        payload: dict[str, Any],
    ) -> RecoveryPrediction:
        """
        Parse a structured provider response.
        """

        if "action" not in payload:
            raise ValueError(
                "Provider response must contain an action."
            )

        try:
            action = RecoveryAction(payload["action"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Unsupported recovery action: {payload.get('action')}"
            ) from exc

        confidence = RecoveryPredictionParser._decimal_or_none(
            payload.get("confidence"),
            field_name="confidence",
        )

        expected_value = RecoveryPredictionParser._decimal_or_none(
            payload.get("expected_value"),
            field_name="expected_value",
        )

        reason = payload.get("reason")

        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                "Provider response reason must be a string."
            )

        policy_metadata = RecoveryPredictionParser._parse_metadata(
            payload.get("policy_metadata"),
        )

        return RecoveryPrediction(
            action=action,
            confidence=confidence,
            expected_value=expected_value,
            reason=reason,
            policy_metadata=policy_metadata,
        )

    @staticmethod
    def _decimal_or_none(
        value: Any,
        *,
        field_name: str,
    ) -> Decimal | None:
        if value is None:
            return None

        try:
            return Decimal(str(value))
        except Exception as exc:
            raise ValueError(
                f"Provider response {field_name} must be numeric."
            ) from exc

    @staticmethod
    def _parse_metadata(
        value: Any,
    ) -> RecoveryPolicyMetadata | None:
        if value is None:
            return None

        if not isinstance(value, dict):
            raise ValueError(
                "Provider response policy_metadata must be an object."
            )

        try:
            policy_name = value["policy_name"]
            policy_version = value["policy_version"]
        except KeyError as exc:
            raise ValueError(
                "Provider response policy_metadata must contain "
                "policy_name and policy_version."
            ) from exc

        return RecoveryPolicyMetadata(
            policy_name=policy_name,
            policy_version=policy_version,
        )