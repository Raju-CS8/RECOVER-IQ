from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from recoveriq.domain.recovery_engine import RecoveryDecision
from recoveriq.domain.recovery_episode import RecoveryEpisodeResult
from recoveriq.domain.recovery_policy_metadata import RecoveryPolicyMetadata
from recoveriq.domain.recovery_scenario import RecoveryScenario


@dataclass(frozen=True, slots=True)
class RecoveryEvaluation:
    """
    Immutable evaluation summary for a completed recovery episode.

    The evaluation converts episode outcomes into business-level recovery
    metrics and preserves policy provenance.
    """

    recovered: bool
    recovered_amount: Decimal

    total_recovery_cost: Decimal
    customer_contact_cost: Decimal
    total_economic_cost: Decimal

    net_recovery_value: Decimal

    decision_count: int
    customer_contact_count: int

    terminal: bool

    policy_metadata: RecoveryPolicyMetadata | None

    @classmethod
    def from_episode(
        cls,
        *,
        episode: RecoveryEpisodeResult,
        scenario: RecoveryScenario,
    ) -> RecoveryEvaluation:
        """
        Build an evaluation summary from a completed recovery episode.
        """

        customer_contact_count = sum(
            1
            for decision in episode.decisions
            if decision.outcome.customer_contacted
        )

        customer_contact_cost = (
            Decimal(customer_contact_count)
            * scenario.customer_contact_cost
        )

        total_economic_cost = (
            episode.total_recovery_cost
            + customer_contact_cost
        )

        net_recovery_value = (
            episode.recovered_amount
            - total_economic_cost
        )

        policy_metadata = cls._resolve_policy_metadata(
            decisions=episode.decisions,
        )

        return cls(
            recovered=episode.recovered,
            recovered_amount=episode.recovered_amount,
            total_recovery_cost=episode.total_recovery_cost,
            customer_contact_cost=customer_contact_cost,
            total_economic_cost=total_economic_cost,
            net_recovery_value=net_recovery_value,
            decision_count=episode.decision_count,
            customer_contact_count=customer_contact_count,
            terminal=episode.terminal,
            policy_metadata=policy_metadata,
        )

    @staticmethod
    def _resolve_policy_metadata(
        *,
        decisions: tuple[RecoveryDecision, ...],
    ) -> RecoveryPolicyMetadata | None:
        """
        Resolve policy provenance from the decisions in the episode.

        All decisions that contain metadata must identify the same policy.
        """

        metadata: RecoveryPolicyMetadata | None = None

        for decision in decisions:
            decision_metadata = decision.policy_metadata

            if decision_metadata is None:
                continue

            if metadata is None:
                metadata = decision_metadata
                continue

            if decision_metadata != metadata:
                raise ValueError(
                    "Recovery episode contains decisions from "
                    "different policies."
                )

        return metadata


class RecoveryEvaluator:
    """
    Application-level evaluator for completed recovery episodes.

    The evaluator calculates business-level recovery economics from an
    episode and its scenario.
    """

    def evaluate(
        self,
        *,
        episode: RecoveryEpisodeResult,
        scenario: RecoveryScenario,
    ) -> RecoveryEvaluation:
        """
        Evaluate a completed recovery episode against its scenario.
        """

        return RecoveryEvaluation.from_episode(
            episode=episode,
            scenario=scenario,
        )