import numpy as np

from agents.baseline_dqn import BaselineDQN
from core.filter_mechanism import FilterMechanism
from core.pulse_salience import PulseSalience
from core.path_protocol import PathProtocol


class MIRAAgent:
    """
    DQN base architecture wrapped with all three MIRA modifications:
      1. FilterMechanism (FILTER) — structured goal tuples with legibility filter
      2. PulseSalience (PULSE) — dynamic salience replaces static weights
      3. PathProtocol (PATH) — controls training loop across phases
    """

    def __init__(self, obs_dim: int, n_actions: int, total_episodes: int = 500,
                 seed: int = 42, **kwargs):
        self.dqn = BaselineDQN(
            obs_dim=obs_dim,
            n_actions=n_actions,
            total_episodes=total_episodes,
            seed=seed,
            **kwargs,
        )
        self.filter_mech = FilterMechanism()
        self.pulse = PulseSalience()
        self.protocol = PathProtocol(self.filter_mech, self.pulse, self.dqn)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.recent_context = []
        self._context_maxlen = 20

    def select_action(self, state: np.ndarray) -> int:
        return self.dqn.select_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.recent_context.append((state, action, reward))
        if len(self.recent_context) > self._context_maxlen:
            self.recent_context.pop(0)
        self.dqn.store_transition(state, action, reward, next_state, done)

    def update(self):
        self.dqn.update()

    def decay_epsilon(self, episode: int):
        self.dqn.decay_epsilon(episode)

    def get_active_goal_id(self) -> str:
        goal = self.filter_mech.get_active_goal()
        if goal is not None:
            return goal["id"]
        return "default_goal"

    def episode_start(self, episode: int, current_state: np.ndarray):
        injected = self.protocol.maybe_inject_scaffolding(
            current_state, list(self.recent_context)
        )
        return injected

    def get_shaped_reward(self, env_reward: float, info: dict) -> float:
        active_goal_id = self.get_active_goal_id()
        phase = self.protocol.get_current_phase()

        scaffolding_bonus = 0.0
        if phase == 2:
            goal_reached = info.get("goal_reached", False)
            loophole_used = info.get("loophole_visits", 0) > 0
            if goal_reached and not loophole_used:
                scaffolding_bonus = 0.5

        return self.protocol.get_reward(env_reward, active_goal_id, scaffolding_bonus)

    def end_episode(self, episode_reward: float, info: dict):
        active_goal_id = self.get_active_goal_id()
        phase = self.protocol.get_current_phase()

        goal_reached = info.get("goal_reached", False)
        pursued = goal_reached or (phase == 2 and active_goal_id != "default_goal")

        current_lambda = self.filter_mech.get_average_lambda()

        if phase >= 2:
            self.pulse.update(active_goal_id, pursued, episode_reward, current_lambda)
            for gid in self.filter_mech.get_all_goal_ids():
                if gid != active_goal_id:
                    self.pulse.update(gid, False, 0.0, current_lambda)

        self.protocol.step_episode(episode_reward)

    def get_phase(self) -> int:
        return self.protocol.get_current_phase()

    def get_average_lambda(self) -> float:
        return self.filter_mech.get_average_lambda()

    def get_top_salience(self, n=3) -> list:
        return self.pulse.get_top_goals(n)

    def get_phase_transitions(self) -> dict:
        return self.protocol.get_phase_transition_episodes()

    def get_internalization_info(self) -> dict:
        """Return internalization depth and gradient for top-3 goals,
        plus the primary (top-1) goal gradient separately."""
        top3 = self.pulse.get_top_goals(3)
        details = []
        for gid, sal in top3:
            details.append({
                "goal_id": gid,
                "salience": sal,
                "internalization_depth": self.pulse.get_internalization_depth(gid),
                "internalization_gradient": self.pulse.get_internalization_gradient(gid, window=20),
            })
        gradients = [d["internalization_gradient"] for d in details]
        return {
            "top3_details": details,
            "internalization_gradient_primary": self.pulse.get_primary_goal_gradient(window=20),
            "internalization_gradient_mean_top3": float(np.mean(gradients)) if gradients else 0.0,
        }

    def get_filter_log(self) -> list:
        return self.filter_mech.get_filter_log()

    def get_transition_log(self) -> list:
        return self.protocol.get_transition_log()

    def is_active_goal_in_top3(self) -> bool:
        active_id = self.get_active_goal_id()
        top3_ids = [gid for gid, _ in self.pulse.get_top_goals(3)]
        return active_id in top3_ids

    def get_active_goal_rank(self) -> int:
        """Return 1-based rank of the active goal among all goals by salience.
        Returns 0 if active goal is 'default_goal' (no goals admitted)."""
        active_id = self.get_active_goal_id()
        if active_id == "default_goal":
            return 0
        ranked = self.pulse.get_top_goals(len(self.pulse.goal_stats))
        for i, (gid, _) in enumerate(ranked):
            if gid == active_id:
                return i + 1
        return 0
