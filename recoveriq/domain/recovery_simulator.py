from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from random import Random


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Deterministic result of a simulated probability trial."""

    successful: bool
    sampled_value: Decimal


class RecoverySimulator:
    """
    Reproducible probability sampler for recovery scenarios.

    The simulator owns randomness. Decision logic must not call random
    directly, which keeps strategy evaluation reproducible and testable.
    """

    def __init__(self, *, seed: int) -> None:
        self._random = Random(seed)

    def trial(self, *, success_probability: Decimal) -> SimulationResult:
        """Run one reproducible Bernoulli trial."""
        probability = Decimal(success_probability)

        if not Decimal("0") <= probability <= Decimal("1"):
            raise ValueError(
                "Success probability must be between zero and one."
            )

        sampled_value = Decimal(str(self._random.random()))

        return SimulationResult(
            successful=sampled_value < probability,
            sampled_value=sampled_value,
        )