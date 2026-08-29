from __future__ import annotations

from recoveriq.domain.recovery_engine import RecoveryEngine
from recoveriq.domain.recovery_environment import RecoveryEnvironment
from recoveriq.domain.recovery_episode import RecoveryEpisode
from recoveriq.domain.recovery_policy import RecoveryDecisionPolicy
from recoveriq.domain.recovery_scenario import RecoveryScenario
from recoveriq.domain.recovery_simulator import RecoverySimulator


class RecoveryWorkflowFactory:
    """
    Construct complete recovery workflows from recovery scenarios.

    The factory centralizes the composition of the policy, environment,
    engine, and episode components required to execute a recovery episode.
    """

    def __init__(
        self,
        *,
        maximum_episode_steps: int,
    ) -> None:
        if maximum_episode_steps <= 0:
            raise ValueError(
                "Maximum episode steps must be greater than zero."
            )

        self._maximum_episode_steps = maximum_episode_steps

    def create_episode(
        self,
        *,
        scenario: RecoveryScenario,
        simulator_seed: int,
    ) -> RecoveryEpisode:
        """
        Create one fully configured recovery episode.

        The supplied scenario configures the policy, environment,
        and episode recovery constraints.

        The simulator seed makes recovery execution deterministic.
        """

        policy = RecoveryDecisionPolicy(
            scenario=scenario,
        )

        simulator = RecoverySimulator(
            seed=simulator_seed,
        )

        environment = RecoveryEnvironment(
            scenario=scenario,
            simulator=simulator,
        )

        engine = RecoveryEngine(
            policy=policy,
            environment=environment,
        )

        return RecoveryEpisode(
            engine=engine,
            scenario=scenario,
            maximum_steps=self._maximum_episode_steps,
        )