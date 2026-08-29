from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from recoveriq.domain.batch_recovery_evaluation import (
    BatchRecoveryEvaluation,
    BatchRecoveryEvaluator,
)
from recoveriq.domain.recovery_episode import RecoveryEpisodeResult
from recoveriq.domain.recovery_evaluation import (
    RecoveryEvaluation,
    RecoveryEvaluator,
)
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_state import RecoveryState


class RecoveryEpisodeRunner(Protocol):
    """Contract required for executing one recovery episode."""

    def run(
        self,
        *,
        initial_state: RecoveryState,
    ) -> RecoveryEpisodeResult:
        ...


class BatchRecoveryEpisodeFactory(Protocol):
    """
    Contract for creating a recovery episode for one batch item.

    Each batch item may have its own recovery scenario and deterministic
    simulator seed.
    """

    def create_episode(
        self,
        *,
        scenario: RecoveryScenario,
        simulator_seed: int,
    ) -> RecoveryEpisodeRunner:
        ...


@dataclass(frozen=True, slots=True)
class BatchRecoveryInput:
    """
    One failed payment to include in a recovery batch.

    Each input retains its own state and recovery scenario because payment
    failures may have different recovery conditions.
    """

    state: RecoveryState
    scenario: RecoveryScenario


@dataclass(frozen=True, slots=True)
class BatchRecoveryWorkflowResult:
    """
    Complete result of a batch recovery workflow.

    Contains individual episode results, business evaluations, and the
    aggregate evaluation across the complete batch.
    """

    episode_results: tuple[RecoveryEpisodeResult, ...]
    evaluations: tuple[RecoveryEvaluation, ...]
    batch_evaluation: BatchRecoveryEvaluation


class BatchRecoveryWorkflow:
    """
    Orchestrates recovery execution for multiple inputs.

    The workflow supports two execution modes:

    - A fixed episode runner, useful for direct orchestration and tests.
    - An episode factory, allowing every batch item to receive its own
      scenario-aware recovery environment and deterministic simulator.

    Exactly one execution strategy must be supplied.
    """

    def __init__(
        self,
        *,
        evaluator: RecoveryEvaluator,
        batch_evaluator: BatchRecoveryEvaluator,
        episode: RecoveryEpisodeRunner | None = None,
        episode_factory: BatchRecoveryEpisodeFactory | None = None,
        simulator_seed: int = 42,
    ) -> None:
        if episode is None and episode_factory is None:
            raise ValueError(
                "Either episode or episode_factory must be provided."
            )

        if episode is not None and episode_factory is not None:
            raise ValueError(
                "Only one of episode or episode_factory may be provided."
            )

        self._episode = episode
        self._episode_factory = episode_factory
        self._evaluator = evaluator
        self._batch_evaluator = batch_evaluator
        self._simulator_seed = simulator_seed

    def run(
        self,
        *,
        inputs: tuple[BatchRecoveryInput, ...],
    ) -> BatchRecoveryWorkflowResult:
        """
        Execute and evaluate every recovery input in the batch.

        When an episode factory is configured, each batch item receives
        an independently configured episode using its own scenario and
        a deterministic seed derived from the batch seed.
        """

        episode_results: list[RecoveryEpisodeResult] = []
        evaluations: list[RecoveryEvaluation] = []

        for index, batch_input in enumerate(inputs):
            episode = self._resolve_episode(
                scenario=batch_input.scenario,
                index=index,
            )

            episode_result = episode.run(
                initial_state=batch_input.state,
            )

            evaluation = self._evaluator.evaluate(
                episode=episode_result,
                scenario=batch_input.scenario,
            )

            episode_results.append(episode_result)
            evaluations.append(evaluation)

        batch_evaluation = self._batch_evaluator.evaluate(
            evaluations=tuple(evaluations),
        )

        return BatchRecoveryWorkflowResult(
            episode_results=tuple(episode_results),
            evaluations=tuple(evaluations),
            batch_evaluation=batch_evaluation,
        )

    def _resolve_episode(
        self,
        *,
        scenario: RecoveryScenario,
        index: int,
    ) -> RecoveryEpisodeRunner:
        """
        Return the episode runner configured for one batch item.
        """

        if self._episode is not None:
            return self._episode

        if self._episode_factory is None:
            raise RuntimeError(
                "Batch recovery workflow has no configured episode runner."
            )

        return self._episode_factory.create_episode(
            scenario=scenario,
            simulator_seed=self._simulator_seed + index,
        )