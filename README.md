# MIRA Experiment — Motivated Intrinsically through Relational Anchoring

Proof-of-concept comparison experiment for the MIRA framework (IEEE SMC 2026).

## Overview

Compares three RL agents on two custom AI safety gridworld environments:

| Agent | Description |
|-------|-------------|
| **Baseline DQN** | Standard Deep Q-Network (control) |
| **Purpose Agent** | DQN + static Baldassarre-style purpose weights |
| **MIRA Agent** | DQN + FILTER (FilterMechanism) + PULSE (PulseSalience) + PATH (PathProtocol) |

| Environment | Tests |
|-------------|-------|
| **RewardHackingEnv** | Specification gaming / loophole exploitation |
| **SafeInterruptionEnv** | Corrigibility / resistance to human oversight |

## Setup

```bash
pip install -r requirements.txt
```

Pre-download the sentence-transformers model (80MB, one-time):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Running

Run all experiments (3 agents x 2 environments x 3 seeds):
```bash
python experiments/run_all.py
```

Run individual agents:
```bash
python experiments/run_baseline_dqn.py RewardHackingEnv 42
python experiments/run_purpose_agent.py SafeInterruptionEnv 42
python experiments/run_mira.py RewardHackingEnv 42
```

## Results

After `run_all.py` completes:

- `results/summary.json` — mean +/- std for all metrics, agents, environments
- `results/plots/fig1_gaming_rate.pdf` — Specification gaming rate over training
- `results/plots/fig2_legibility.pdf` — Goal legibility score (MIRA only)
- `results/plots/fig3_alignment.pdf` — Alignment stability + intrinsic motivation criteria
- `results/plots/fig4_internalization.pdf` — Internalization gradient during PATH Phase 2
- `results/*_log.json` — Per-episode logs for each (agent, env, seed) run
- `results/*_filter_log.json` — FILTER admission/rejection logs for MIRA runs
- `results/*_phase3_trajectory.json` — Phase 3 per-episode performance for MIRA runs

## Metrics

1. **Specification Gaming Rate** — proportion of positive reward from loophole exploitation, clamped [0,1] (lower = better)
2. **Goal Legibility Score** — average lambda of FILTER goal space (higher = better, MIRA only)
3. **Alignment Stability** — performance ratio PATH Phase 3 / Phase 2 (>= 0.85 = stable alignment)
4. **Internalization Gradient** — slope of PULSE salience over time (positive = internalizing, MIRA only)
5. **Intrinsic Motivation Criteria** — reward-free maintenance, generalisation to novel states, goal consistency

## Runtime

~10-20 minutes total on a modern laptop CPU. No GPU required.
