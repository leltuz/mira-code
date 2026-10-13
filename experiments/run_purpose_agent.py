"""Run Purpose Agent (Baldassarre-style) on a given environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm

from environments import RewardHackingEnv, SafeInterruptionEnv
from agents.purpose_agent import PurposeAgent


def run_purpose_agent(env_name: str, n_episodes: int = 500, seed: int = 42) -> list:
    if env_name == "RewardHackingEnv":
        env = RewardHackingEnv()
    elif env_name == "SafeInterruptionEnv":
        env = SafeInterruptionEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = PurposeAgent(obs_dim=obs_dim, n_actions=n_actions,
                         total_episodes=n_episodes, seed=seed)

    logs = []
    for ep in tqdm(range(n_episodes), desc=f"PurposeAgent/{env_name}/seed={seed}"):
        state, info = env.reset(seed=seed + ep)
        agent.decay_epsilon(ep)
        agent.reset_episode_counters()
        total_reward = 0.0
        total_shaped_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            shaped_reward = agent.shape_reward(reward, info)
            agent.store_transition(state, action, shaped_reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            total_shaped_reward += shaped_reward

        log_entry = {
            "episode": ep,
            "total_reward": total_reward,
            "total_shaped_reward": total_shaped_reward,
            "performance": info.get("performance", 0.0),
            "goal_reached": info.get("goal_reached", False),
            "loophole_visits": info.get("loophole_visits", 0),
            "loophole_reward": info.get("loophole_reward", 0.0),
            "goal_reward": info.get("goal_reward", 0.0),
            "i_zone_visits": info.get("i_zone_visits", 0),
            "interrupt_count": info.get("interrupt_count", 0),
            "resistance_score": info.get("resistance_score", 0.0),
            "episode_length": env.steps if hasattr(env, "steps") else 0,
            "purpose_weights": agent.get_purpose_weights(),
            "legibility_score": 0.0,
            "has_filter_mechanism": False,
        }
        logs.append(log_entry)

    return logs


if __name__ == "__main__":
    env_name = sys.argv[1] if len(sys.argv) > 1 else "RewardHackingEnv"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    logs = run_purpose_agent(env_name, seed=seed)
    print(f"Final avg reward (last 50): {np.mean([l['total_reward'] for l in logs[-50:]]):.2f}")
