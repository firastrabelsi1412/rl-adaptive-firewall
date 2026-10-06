"""
Phase 4 — Gymnasium environment for the PPO firewall agent.

Episode structure
-----------------
**Contextual bandit: 1 window = 1 episode.** Each call to ``reset()`` samples
a random sliding window from the dataset, extracts its 256-dim BiLSTM state
via the frozen v1 detector, and returns it as the observation. ``step(action)``
computes the reward from the action × true-class table, then immediately
returns ``terminated=True``. PPO sees ``done=True`` after every step, so the
discount factor ``gamma`` collapses out of the advantage calculation — the
agent is learning a one-shot policy, which matches the §11.4 production
semantics (each flow window is an independent ALLOW/BLOCK/RATE-LIMIT/LOG
decision).

Observation space
-----------------
``Box(low=-inf, high=+inf, shape=(256,), dtype=float32)``. Honest unbounded
spec — the BiLSTM ``out[:, -1, :]`` tensor isn't strictly tanh-clipped (it's
post-tanh on the cell state but pre-projection through the classifier head,
which we don't apply here).

Action space
------------
``Discrete(4)``: ``{0:ALLOW, 1:BLOCK, 2:RATE_LIMIT, 3:LOG}``. Mirrors the
§11.4 PPO-action → iptables-target mapping verbatim.

Reward design
-------------
The §7 Phase 4 spec calls for ``+1`` for correctly blocking a threat, ``-1``
for blocking benign traffic, and a ``-0.5`` latency penalty per action. The
4×5 table below extends that across all (action × class) cells; the latency
penalty is then subtracted uniformly from every cell, so each call to ``step``
returns ``REWARD_TABLE[action, true_class] - LATENCY_PENALTY``.

Bold-equivalent cells (the optimal action per class) are:

* BLOCK on any attack class: ``+1.0 - 0.5 = +0.5``
* ALLOW or LOG on Normal: ``0.0 - 0.5 = -0.5``

A trained policy that always picks the class-optimal action would average
roughly ``+0.07`` per step on the NSL-KDD distribution (Normal ≈43%, attacks
≈57%); a uniform-random policy averages roughly ``-0.58``. So a positive
``rollout/ep_rew_mean`` in TensorBoard means PPO has learned to discriminate
benign-vs-attack on the LSTM-CNN features.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.agent.feature_extractor import FrozenLSTMCNN


class Action(IntEnum):
    ALLOW      = 0
    BLOCK      = 1
    RATE_LIMIT = 2
    LOG        = 3


# Reward grid: rows = action (0..3), cols = true class (0..4 = N/DoS/Probe/R2L/U2R).
# The latency penalty is NOT baked in here — it's subtracted in step() so the
# table reads identically to the §7 Phase 4 spec table.
REWARD_TABLE: np.ndarray = np.array(
    #   Normal   DoS   Probe   R2L    U2R
    [
        [  0.0, -1.0,  -1.0, -1.0,  -1.0],   # ALLOW       — 0 on benign, -1 on missed detection
        [ -1.0, +1.0,  +1.0, +1.0,  +1.0],   # BLOCK       — -1 on FP, +1 on threat blocked
        [ -0.3, +0.3,  +0.3, +0.3,  +0.3],   # RATE_LIMIT  — partial credit/penalty (compromise action)
        [  0.0, +0.1,  +0.1, +0.1,  +0.1],   # LOG         — informational, free on benign
    ],
    dtype=np.float32,
)

LATENCY_PENALTY: float = 0.5

# Inferred from FrozenLSTMCNN class attrs but copied here as module-level
# constants so the env can be defined without importing torch at module scope
# in test contexts.
T_WINDOW:    int = FrozenLSTMCNN.T_WINDOW    # 10
N_FEATURES:  int = FrozenLSTMCNN.N_FEATURES  # 41
FEATURE_DIM: int = FrozenLSTMCNN.FEATURE_DIM # 256

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


class FirewallEnv(gym.Env):
    """Contextual-bandit Gymnasium environment over LSTM-CNN feature windows.

    Parameters
    ----------
    X : np.ndarray
        Pre-windowed inputs, shape ``(N, T_WINDOW, N_FEATURES) = (N, 10, 41)``,
        dtype ``float32``. The same ``X_train.npy`` / ``X_test.npy`` produced
        by Phase 2.
    y : np.ndarray
        5-class labels, shape ``(N,)``, dtype integer, values in
        ``{0,1,2,3,4}``. The caller is responsible for applying
        ``src.preprocessing.labels.remap_labels`` *before* construction —
        keeping the 40→5 remap as a one-shot at the data-loading boundary
        (rather than per-step) avoids re-running ``np.vectorize`` at every
        ``reset()``.
    feature_extractor : FrozenLSTMCNN
        The shared frozen v1 detector (256-dim BiLSTM output). One instance
        is shared across all parallel envs — it has no internal state, no
        gradients, and no per-call buffers, so concurrent forward calls are
        safe under SB3's ``DummyVecEnv`` (which is single-threaded anyway).

    Notes
    -----
    *RNG.* ``self.np_random`` is initialised by ``super().reset(seed=...)``
    to a ``np.random.Generator`` (Gym 0.26+). All sampling uses this
    generator — no legacy ``RandomState``. Each parallel env gets its own
    seed (``42 + worker_id``) so the 4 envs see disjoint random sequences.

    *Dtype hygiene.* Inputs are cast to ``torch.float32`` before the forward
    pass; the returned observation is asserted ``np.float32`` to match the
    Box space. Catches dtype drift at the env boundary instead of inside SB3.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_extractor: FrozenLSTMCNN,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # ── Input validation ─────────────────────────────────────────────────
        if X.ndim != 3 or X.shape[1] != T_WINDOW or X.shape[2] != N_FEATURES:
            raise ValueError(
                f"X must have shape (N, {T_WINDOW}, {N_FEATURES}); got {X.shape}"
            )
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"y must have shape ({X.shape[0]},); got {y.shape}"
            )
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"y must be integer-typed; got {y.dtype}")
        unique = np.unique(y)
        if not set(unique.tolist()).issubset({0, 1, 2, 3, 4}):
            raise ValueError(
                f"y must contain only class ids in [0,4]; got {unique.tolist()}. "
                f"Did you forget to apply src.preprocessing.labels.remap_labels?"
            )

        # Cast X to float32 once at construction (cheap, eliminates per-reset
        # re-cast and dtype mismatch warnings).
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        self._X = X
        self._y = y.astype(np.int64, copy=False)
        self._n_windows = int(X.shape[0])
        self._extractor = feature_extractor
        self._initial_seed = seed

        # ── Spaces ───────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(Action))   # 4

        # ── Episode state ────────────────────────────────────────────────────
        self._current_state: np.ndarray | None = None
        self._current_label: int | None = None
        self._last_idx:      int | None = None
        self._last_action:   int | None = None
        self._last_reward:   float | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Pass the construction-time seed on the very first reset if none was
        # explicitly given — keeps the 4 parallel envs reproducible across runs.
        if seed is None and self._last_idx is None and self._initial_seed is not None:
            seed = self._initial_seed
        super().reset(seed=seed)   # populates self.np_random as np.random.Generator

        idx = int(self.np_random.integers(0, self._n_windows))
        window_np = self._X[idx]                                        # (10, 41)
        window_t  = torch.from_numpy(window_np).unsqueeze(0)            # (1, 10, 41)
        if window_t.dtype != torch.float32:
            window_t = window_t.to(torch.float32)

        state_t  = self._extractor(window_t)                            # (1, 256)
        state_np = state_t[0].detach().cpu().numpy().astype(np.float32, copy=False)

        # Dtype hygiene assertion — matches Box dtype, catches silent drift.
        assert state_np.dtype == np.float32, f"state dtype {state_np.dtype} != float32"
        assert state_np.shape == (FEATURE_DIM,), f"state shape {state_np.shape} != ({FEATURE_DIM},)"

        self._current_state = state_np
        self._current_label = int(self._y[idx])
        self._last_idx      = idx
        self._last_action   = None
        self._last_reward   = None

        return state_np, {"true_class": self._current_label, "idx": idx}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # SB3 may pass numpy scalars; coerce to int for indexing + dict keys.
        action_int = int(action)
        if not 0 <= action_int < len(Action):
            raise ValueError(f"action {action_int} not in [0, {len(Action)})")
        if self._current_state is None or self._current_label is None:
            raise RuntimeError("step() called before reset()")

        reward = float(REWARD_TABLE[action_int, self._current_label] - LATENCY_PENALTY)

        self._last_action = action_int
        self._last_reward = reward

        # Contextual bandit: every step terminates the episode.
        terminated = True
        truncated  = False
        info = {
            "true_class":      self._current_label,
            "true_class_name": CLASS_NAMES[self._current_label],
            "action":          action_int,
            "action_name":     Action(action_int).name,
            "reward":          reward,
        }
        return self._current_state, reward, terminated, truncated, info

    def render(self) -> None:
        if self._last_idx is None:
            print("FirewallEnv(uninitialised — call reset() first)")
            return
        cls_name = CLASS_NAMES[self._current_label] if self._current_label is not None else "?"
        act_name = Action(self._last_action).name if self._last_action is not None else "—"
        rwd      = f"{self._last_reward:+.2f}" if self._last_reward is not None else "—"
        print(
            f"FirewallEnv(idx={self._last_idx:>6d}  "
            f"true_class={cls_name:<7s}  "
            f"last_action={act_name:<11s}  "
            f"last_reward={rwd})"
        )
