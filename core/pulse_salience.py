import numpy as np


class PulseSalience:
    """
    PULSE: Maintains a dynamic salience score rho(g, t) for each goal in the
    FILTER goal space. Replaces static purpose weights with a live function
    of interaction history. In PATH Phase 3, this salience score IS the reward
    signal. Also serves as an operational proxy for internalization depth.

    rho(g, t) = alpha * co_construction_freq(g)
              + beta  * interaction_centrality(g)
              + gamma * legibility_stability(g)
    """

    def __init__(self, alpha=0.4, beta=0.35, gamma=0.25):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.goal_stats = {}
        self._salience_history = {}

    def update(self, goal_id: str, pursued: bool,
               episode_reward: float, current_lambda: float):
        if goal_id not in self.goal_stats:
            self.goal_stats[goal_id] = {
                "pursuit_count": 0,
                "total_episodes": 0,
                "reward_history": [],
                "lambda_history": [],
            }
        stats = self.goal_stats[goal_id]
        stats["total_episodes"] += 1
        if pursued:
            stats["pursuit_count"] += 1
            stats["reward_history"].append(episode_reward)
        stats["lambda_history"].append(current_lambda)

        salience = self.compute_salience(goal_id)
        if goal_id not in self._salience_history:
            self._salience_history[goal_id] = []
        self._salience_history[goal_id].append(salience)

    def compute_salience(self, goal_id: str) -> float:
        if goal_id not in self.goal_stats:
            return 0.0
        stats = self.goal_stats[goal_id]

        if stats["total_episodes"] == 0:
            freq = 0.0
        else:
            freq = stats["pursuit_count"] / stats["total_episodes"]

        if not stats["reward_history"]:
            centrality = 0.0
        else:
            raw = np.mean(stats["reward_history"])
            centrality = 1.0 / (1.0 + np.exp(-raw / 10.0))

        if len(stats["lambda_history"]) < 2:
            stability = 1.0
        else:
            stability = max(0.0, 1.0 - np.std(stats["lambda_history"]))

        return float(self.alpha * freq + self.beta * centrality + self.gamma * stability)

    def get_top_goals(self, n=3) -> list:
        scored = [(gid, self.compute_salience(gid)) for gid in self.goal_stats]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:n]

    def get_internalization_depth(self, goal_id: str) -> float:
        """
        A salience score of 0.0 indicates no internalization (external regulation).
        A score approaching 1.0 indicates deep internalization
        (functionally intrinsic motivation).
        """
        return self.compute_salience(goal_id)

    def get_internalization_gradient(self, goal_id: str, window: int = 20) -> float:
        """
        Compute the linear slope of salience scores over the last [window]
        episodes for the given goal. A positive slope indicates active
        internalization.
        """
        history = self._salience_history.get(goal_id, [])
        if len(history) < 2:
            return 0.0
        effective_window = max(min(window, len(history) - 1), 2)
        segment = history[-effective_window:]
        x = np.arange(len(segment), dtype=float)
        y = np.array(segment, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def get_primary_goal_gradient(self, window=20) -> float:
        """
        Returns the internalization gradient of the single highest-salience
        goal only. A positive value indicates the primary goal is being
        actively internalized.
        """
        if not self.goal_stats:
            return 0.0
        top_goal_id = max(
            self.goal_stats.keys(),
            key=lambda gid: self.compute_salience(gid)
        )
        return self.get_internalization_gradient(top_goal_id, window=window)

    def salience_as_reward(self, goal_id: str, base_reward: float) -> float:
        """
        In PATH Phase 3: replace external reward with salience-weighted signal.
        base_reward is passed for normalization reference but the return
        value is driven by PULSE salience, plus a small floor for
        training stability.
        """
        salience = self.compute_salience(goal_id)
        return salience * 10.0 + 0.01
