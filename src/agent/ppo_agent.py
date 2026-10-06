"""
Phase 4 — PPO agent configuration wrapper.

Thin layer over ``stable_baselines3.PPO``: holds the hyperparameter dict and
exposes ``build_train_env`` / ``build_eval_env`` / ``build_agent`` factories
so ``training/train_ppo.py`` and ``evaluation/evaluate_ppo.py`` can share the
same construction logic.

No custom RL code lives here — SB3 owns the rollout, the GAE, and the PPO
clipped-surrogate update. This module exists to keep configuration in one
place so a hyperparameter sweep changes one file, not two.

Hyperparameter rationale (see STEP 2 plan in conversation, §7 of project doc)
---------------------------------------------------------------------------
* ``MlpPolicy``                — 256-dim flat observation, no spatial structure.
* ``learning_rate=3e-4``       — SB3 default; matches Phase 3 LSTM-CNN LR.
* ``n_steps=2048``             — rollout length per env; 4 envs → 8192 samples
                                 per PPO update.
* ``batch_size=64``            — minibatch for the PPO update; 128 minibatches
                                 per epoch.
* ``n_epochs=10``              — number of full passes over the 8192-sample
                                 rollout per update.
* ``gamma=0.99``               — irrelevant for the 1-step contextual-bandit
                                 framing (every episode ends after one step),
                                 but SB3 requires a value.
* ``ent_coef=0.01``            — slightly above the SB3 default of 0.0. Keeps
                                 the policy stochastic across all 4 actions
                                 long enough to prevent the always-BLOCK
                                 collapse warned about in STEP 2(E).
* ``max_grad_norm=0.5``        — SB3 default; insurance against any NaN/inf
                                 in the policy gradient.
* ``seed=42``                  — reproducibility anchor; the parallel envs
                                 receive seeds 42..45 separately.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.agent.feature_extractor import FrozenLSTMCNN
from src.environment.firewall_env import FirewallEnv


PPO_CONFIG: dict[str, Any] = {
    "policy":         "MlpPolicy",
    "learning_rate":  3e-4,
    "n_steps":        2048,
    "batch_size":     64,
    "n_epochs":       10,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "ent_coef":       0.01,
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "seed":           42,
    "verbose":        0,
}


# ── Vec-env factories ────────────────────────────────────────────────────────


def _make_env_thunk(
    X: np.ndarray,
    y: np.ndarray,
    extractor: FrozenLSTMCNN,
    seed: int,
) -> Callable[[], Monitor]:
    """Return a zero-arg factory that builds one Monitor-wrapped FirewallEnv.

    The double-closure pattern (factory returning factory) is required by
    DummyVecEnv / SubprocVecEnv — both accept ``Callable[[], gym.Env]``.

    The ``Monitor`` wrapper is required for SB3 to populate
    ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean`` in TensorBoard via
    the internal ``ep_info_buffer``. Without it, the reward curve is silently
    missing — the main signal we need to know if PPO is learning.
    """
    def _build() -> Monitor:
        return Monitor(FirewallEnv(X, y, extractor, seed=seed))
    return _build


def build_train_env(
    X: np.ndarray,
    y: np.ndarray,
    extractor: FrozenLSTMCNN,
    n_envs: int = 4,
    base_seed: int = 42,
) -> VecEnv:
    """Construct an ``n_envs``-way DummyVecEnv with disjoint per-env seeds.

    DummyVecEnv (single-process, sequential) is chosen over SubprocVecEnv
    because:

    1. The shared frozen feature extractor would be ``pickle``-copied into
       every subprocess, blowing up memory needlessly.
    2. Windows ``multiprocessing.spawn`` interacts poorly with the Phase 3
       training script's ``num_workers=0`` discipline; staying single-process
       avoids the entire problem class.
    3. The bottleneck is the LSTM-CNN forward (CPU-bound, 256-dim output),
       not policy inference — parallelism wouldn't help unless we had a GPU.
    """
    return DummyVecEnv(
        [_make_env_thunk(X, y, extractor, base_seed + i) for i in range(n_envs)]
    )


def build_eval_env(
    X: np.ndarray,
    y: np.ndarray,
    extractor: FrozenLSTMCNN,
    seed: int = 999,
) -> VecEnv:
    """Single-env DummyVecEnv for the EvalCallback.

    EvalCallback expects a VecEnv with the same observation/action space as
    the training env. n_envs=1 here so the eval reward is reported as a
    single scalar mean rather than averaged across heterogeneous parallel
    streams.
    """
    return DummyVecEnv([_make_env_thunk(X, y, extractor, seed)])


# ── Agent factory ────────────────────────────────────────────────────────────


def build_agent(
    env: VecEnv,
    tb_log_dir: str | None = None,
    **overrides: Any,
) -> PPO:
    """Construct a ``stable_baselines3.PPO`` with the shared config.

    Parameters
    ----------
    env : VecEnv
        Already-vectorised training env (typically from ``build_train_env``).
    tb_log_dir : str or None
        Path for TensorBoard logs. Pass ``None`` to disable TB logging.
    **overrides
        Any key from ``PPO_CONFIG`` can be overridden at call time — useful
        for smoke tests (e.g. ``n_steps=256`` to fire an update sooner).
    """
    cfg = {**PPO_CONFIG, **overrides}
    if tb_log_dir is not None:
        cfg["tensorboard_log"] = tb_log_dir
    return PPO(env=env, **cfg)
