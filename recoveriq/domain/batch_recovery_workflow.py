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


@dataclass(frozen=True, slots=True)
class BatchRecoveryInput:
    """
    One recovery item in a batch.

    Each item contains its own recovery state and scenario because
    different failed payments may have different recovery conditions.
    """

    state: RecoveryState
    scenario: RecoveryScenario


@dataclass(frozen=True, slots=True)
class BatchRecoveryWorkflowResult:
    """
    Complete result of a batch recovery workflow.

    Contains individual episode results, their business evaluations,
    and the final aggregate batch evaluation.
    """

    episode_results: tuple[RecoveryEpisodeResult, ...]
    evaluations: tuple[RecoveryEvaluation, ...]
    batch_evaluation: BatchRecoveryEvaluation


class BatchRecoveryWorkflow:
    """
    Orchestrates recovery execution and evaluation for multiple inputs.

    The workflow does not make recovery decisions itself. It coordinates
    an episode runner, individual evaluator, and batch evaluator.
    """

    def __init__(
        self,
        *,
        episode: RecoveryEpisodeRunner,
        evaluator: RecoveryEvaluator,
        batch_evaluator: BatchRecoveryEvaluator,
    ) -> None:
        self._episode = episode
        self._evaluator = evaluator
        self._batch_evaluator = batch_evaluator

    def run(
        self,
        *,
        inputs: tuple[BatchRecoveryInput, ...],
    ) -> BatchRecoveryWorkflowResult:
        """
        Execute and evaluate every recovery input in the batch.
        """

        episode_results: list[RecoveryEpisodeResult] = []
        evaluations: list[RecoveryEvaluation] = []

        for batch_input in inputs:
            episode_result = self._episode.run(
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