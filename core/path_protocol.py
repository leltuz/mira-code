import numpy as np

from core.filter_mechanism import FilterMechanism, SCAFFOLDING_GOALS
from core.pulse_salience import PulseSalience


class PathProtocol:
    """
    PATH: Controls the training loop across three phases.
    Phase transitions are governed by explicit criteria.

    Phase 1 (0-99):    Pre-relational bootstrapping — standard RL reward
    Phase 2 (100-249):  Vygotskian scaffolding — RL reward + scaffolding bonus
    Phase 3 (200+-499): Autonomous relational operation — PULSE salience only

    Phase 2 → 3 requires: goal space >= 5, avg lambda >= 0.40,
    AND internalization gradient positive for >= 2 of top-3 goals.
    If not met by episode 250, transition forced with a warning.
    """

    PHASE2_HARD_CAP = 250

    def __init__(self, filter_mech: FilterMechanism,
                 pulse: PulseSalience, agent):
        self.filter_mech = filter_mech
        self.pulse = pulse
        self.agent = agent
        self.current_phase = 1
        self.episode = 0
        self.phase_transition_episodes = {}
        self.recent_rewards = []
        self.phase2_start_episode = None
        self.scaffolding_log = []
        self.forced_transition = False
        self.transition_log = []

    def get_current_phase(self) -> int:
        return self.current_phase

    def _check_internalization_criterion(self) -> bool:
        """Check if >= 2 of top-3 goals have positive internalization gradient."""
        top3 = self.pulse.get_top_goals(3)
        if len(top3) < 2:
            return False
        positive_count = sum(
            1 for gid, _ in top3
            if self.pulse.get_internalization_gradient(gid, window=20) > 0
        )
        return positive_count >= 2

    def should_transition(self) -> bool:
        if self.current_phase == 1:
            if (self.episode >= 100
                    and len(self.recent_rewards) >= 20
                    and np.mean(self.recent_rewards[-20:]) > -5.0):
                return True
        elif self.current_phase == 2:
            base_met = (
                self.filter_mech.get_goal_space_size() >= 5
                and self.filter_mech.get_average_lambda() >= 0.40
            )
            if self.episode >= 200 and base_met:
                if self._check_internalization_criterion():
                    return True
            if self.episode >= self.PHASE2_HARD_CAP and base_met:
                self.forced_transition = True
                print(
                    "PATH: Phase 2→3 transition forced at episode limit. "
                    "Internalization gradient criterion not satisfied. "
                    "This episode will be flagged in results."
                )
                return True
        return False

    def transition_phase(self):
        log_entry = {
            "from_phase": self.current_phase,
            "to_phase": self.current_phase + 1,
            "episode": self.episode,
            "forced_transition": self.forced_transition,
        }
        self.transition_log.append(log_entry)
        self.phase_transition_episodes[self.current_phase] = self.episode
        self.current_phase += 1
        if self.current_phase == 2:
            self.phase2_start_episode = self.episode
        self.forced_transition = False

    def get_reward(self, env_reward: float, active_goal_id: str,
                   scaffolding_bonus: float = 0.0) -> float:
        if self.current_phase == 1:
            return env_reward
        elif self.current_phase == 2:
            return env_reward + scaffolding_bonus
        else:
            return self.pulse.salience_as_reward(active_goal_id, env_reward)

    def maybe_inject_scaffolding(self, current_state: np.ndarray,
                                 recent_context: list) -> dict | None:
        if self.current_phase != 2 or self.phase2_start_episode is None:
            return None

        offset = self.episode - self.phase2_start_episode
        for sched_offset, description in SCAFFOLDING_GOALS:
            if offset == sched_offset:
                goal = self.filter_mech.propose_goal(
                    current_state, recent_context, description,
                    path_phase=self.current_phase,
                )
                self.scaffolding_log.append({
                    "episode": self.episode,
                    "description": description,
                    "admitted": goal is not None,
                    "lambda": goal["lambda"] if goal else None,
                })
                if goal is not None:
                    goal["admitted_at_episode"] = self.episode
                return goal
        return None

    def step_episode(self, episode_reward: float):
        self.recent_rewards.append(episode_reward)
        self.episode += 1
        if self.should_transition() and self.current_phase < 3:
            self.transition_phase()

    def get_phase_transition_episodes(self) -> dict:
        return dict(self.phase_transition_episodes)

    def get_transition_log(self) -> list:
        return list(self.transition_log)
