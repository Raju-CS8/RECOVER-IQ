from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from recoveriq.domain.recovery_episode import RecoveryEpisodeResult
from recoveriq.domain.recovery_state import RecoveryState


class RecoveryEpisodeRunner(Protocol):
    """
    Minimal contract required by the batch runner.

    A configured recovery episode implementation runs one recovery process
    for a supplied initial state.
    """

    def run(
        self,
        *,
        initial_state: RecoveryState,
    ) -> RecoveryEpisodeResult:
        ...


@dataclass(frozen=True, slots=True)
class BatchRecoveryResult:
    """
    Immutable result of running recovery episodes for multiple states.
    """

    episodes: tuple[RecoveryEpisodeResult, ...]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)


class BatchRecoveryRunner:
    """
    Executes a configured recovery episode for multiple initial states.

    The runner performs orchestration only. Recovery decisions and episode
    execution remain the responsibility of the configured episode.
    """

    def __init__(
        self,
        *,
        episode: RecoveryEpisodeRunner,
    ) -> None:
        self._episode = episode

    def run(
        self,
        *,
        initial_states: tuple[RecoveryState, ...],
    ) -> BatchRecoveryResult:
        """
        Run the configured recovery episode for every supplied state.
        """

        episodes = tuple(
            self._episode.run(
                initial_state=state,
            )
            for state in initial_states
        )

        return BatchRecoveryResult(
            episodes=episodes,
        )