from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecoveryPolicyMetadata:
    """
    Immutable identity and version metadata for a recovery policy.

    This metadata identifies the decision strategy that produced a
    prediction. It does not contain the prediction itself.
    """

    policy_name: str
    policy_version: str

    def __post_init__(self) -> None:
        if not self.policy_name.strip():
            raise ValueError(
                "Recovery policy name must not be empty."
            )

        if not self.policy_version.strip():
            raise ValueError(
                "Recovery policy version must not be empty."
            )