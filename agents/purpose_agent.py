import numpy as np

from agents.baseline_dqn import BaselineDQN


PURPOSES = {
    "reach_goal":       {"weight": 1.0, "bonus": 2.0},
    "avoid_loophole":   {"weight": 0.5, "bonus": -1.0},
    "accept_interrupt":  {"weight": 0.3, "bonus": 1.0},
}


class PurposeAgent(BaselineDQN):
    """
    Same DQN architecture as BaselineDQN, but with static Baldassarre-style
    purpose weights applied as a reward shaping layer.
    Weights are fixed throughout training — key contrast with MIRA.
    """

    def __init__(self, obs_dim: int, n_actions: int, **kwargs):
        super().__init__(obs_dim, n_actions, **kwargs)
        self.purposes = PURPOSES
        self.active_purpose_weights = {
            k: v["weight"] for k, v in PURPOSES.items()
        }

    def shape_reward(self, base_reward: float, info: dict) -> float:
        shaped = base_reward

        if info.get("goal_reached", False):
            shaped += self.purposes["reach_goal"]["weight"] * self.purposes["reach_goal"]["bonus"]

        if info.get("loophole_visits", 0) > 0:
            current_visits = info.get("loophole_visits", 0)
            if not hasattr(self, "_prev_loophole_visits"):
                self._prev_loophole_visits = 0
            if current_visits > self._prev_loophole_visits:
                shaped += self.purposes["avoid_loophole"]["weight"] * self.purposes["avoid_loophole"]["bonus"]
                self._prev_loophole_visits = current_visits

        if info.get("i_zone_visits", 0) > 0:
            current_iz = info.get("i_zone_visits", 0)
            if not hasattr(self, "_prev_i_zone_visits"):
                self._prev_i_zone_visits = 0
            if current_iz > self._prev_i_zone_visits:
                shaped += self.purposes["accept_interrupt"]["weight"] * self.purposes["accept_interrupt"]["bonus"]
                self._prev_i_zone_visits = current_iz

        return shaped

    def reset_episode_counters(self):
        self._prev_loophole_visits = 0
        self._prev_i_zone_visits = 0

    def get_purpose_weights(self) -> dict:
        return dict(self.active_purpose_weights)
