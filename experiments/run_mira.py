"""Run MIRA Agent (FILTER + PULSE + PATH) on a given environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random as stdlib_random
import numpy as np
from tqdm import tqdm

from environments import RewardHackingEnv, SafeInterruptionEnv
from agents.mira_agent import MIRAAgent


NOVEL_STARTS_RH = [(0, 7), (7, 0), (3, 3), (6, 1), (1, 6)]
ALL_STARTS_RH = [(0, 0)] + NOVEL_STARTS_RH


def run_mira(env_name: str, n_episodes: int = 500, seed: int = 42) -> list:
    if env_name == "RewardHackingEnv":
        env = RewardHackingEnv()
    elif env_name == "SafeInterruptionEnv":
        env = SafeInterruptionEnv()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = MIRAAgent(obs_dim=obs_dim, n_actions=n_actions,
                      total_episodes=n_episodes, seed=seed)

    rng = stdlib_random.Random(seed)
    logs = []

    for ep in tqdm(range(n_episodes), desc=f"MIRAAgent/{env_name}/seed={seed}"):
        phase = agent.get_phase()
        novel_start = False
        reset_options = None

        if (env_name == "RewardHackingEnv"
                and phase == 3
                and ep >= n_episodes - 50):
            start_pos = rng.choice(ALL_STARTS_RH)
            reset_options = {"start_pos": start_pos}

        state, info = env.reset(seed=seed + ep, options=reset_options)

        actual_start = (0, 0)
        if hasattr(env, '_start_pos'):
            actual_start = tuple(env._start_pos)
            novel_start = (actual_start != (0, 0))
        agent.decay_epsilon(ep)

        injected = agent.episode_start(ep, state)

        total_reward = 0.0
        total_shaped_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward = agent.get_shaped_reward(reward, info)
            agent.store_transition(state, action, shaped_reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            total_shaped_reward += shaped_reward

        agent.end_episode(total_reward, info)

        phase = agent.get_phase()
        avg_lam = agent.get_average_lambda()
        top_salience = agent.get_top_salience(3)
        intern_info = agent.get_internalization_info()
        goal_in_top3 = agent.is_active_goal_in_top3()
        goal_rank = agent.get_active_goal_rank()

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
            "path_phase": phase,
            "avg_lambda": avg_lam,
            "top_salience": top_salience,
            "phase_transitions": agent.get_phase_transitions(),
            "has_filter_mechanism": True,
            "legibility_score": avg_lam,
            "internalization_gradient_primary": intern_info["internalization_gradient_primary"],
            "internalization_gradient_mean_top3": intern_info["internalization_gradient_mean_top3"],
            "internalization_top3": intern_info["top3_details"],
            "active_goal_in_top3": goal_in_top3,
            "active_goal_rank": goal_rank,
            "novel_start": novel_start,
            "start_position": list(actual_start),
        }
        logs.append(log_entry)

    filter_log = agent.get_filter_log()
    return logs, filter_log


if __name__ == "__main__":
    env_name = sys.argv[1] if len(sys.argv) > 1 else "RewardHackingEnv"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    logs, filter_log = run_mira(env_name, seed=seed)
    print(f"Final avg reward (last 50): {np.mean([l['total_reward'] for l in logs[-50:]]):.2f}")
    print(f"Phase transitions: {logs[-1].get('phase_transitions', {})}")
