"""
Phase 4 — PPO firewall agent training script.

Loads the frozen v1 LSTM-CNN detector (§21 confirmed this is the right
checkpoint; §15.3 frozen-benchmark gate vetoed both v2 attempts), wraps a
``FirewallEnv`` around its 256-dim BiLSTM output, and trains a PPO agent
for ``TOTAL_TIMESTEPS`` steps. Best-by-eval-mean-reward checkpoint is saved
to ``data/models/ppo_best.zip``.

Override the step count for smoke testing via the ``PPO_TOTAL_STEPS`` env
var (defaults to 500_000). All other config lives in ``src.agent.ppo_agent``.

Usage
-----
    # Full run
    python training/train_ppo.py 2>&1 | tee data/processed/_ppo_train_log.txt

    # 20k smoke
    $env:PPO_TOTAL_STEPS = "20000"
    python training/train_ppo.py
"""

from __future__ import annotations

# ── sys.path injection ────────────────────────────────────────────────────────
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Imports ───────────────────────────────────────────────────────────────────
import shutil
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from config import PROC_DIR, MODELS_DIR
from src.agent.feature_extractor import FrozenLSTMCNN
from src.agent.ppo_agent import (
    PPO_CONFIG,
    build_agent,
    build_eval_env,
    build_train_env,
)
from src.environment.firewall_env import CLASS_NAMES, Action
from src.preprocessing.labels import remap_labels


# ── Constants ─────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS: int = int(os.environ.get("PPO_TOTAL_STEPS", 500_000))

# EvalCallback fires when n_calls (vec-env steps) % EVAL_FREQ == 0. With 4
# parallel envs, EVAL_FREQ=5_000 vec-env steps = 20_000 transitions → ~25
# eval checkpoints over a 500k-step run.
EVAL_FREQ:       int = 5_000
N_EVAL_EPISODES: int = 1_000     # 1k episodes/eval × 25 evals × 1-step bandit = 25k total eval steps

N_ENVS:    int = 4
BASE_SEED: int = 42
EVAL_SEED: int = 999

# Output paths
TB_LOG_DIR:     str = os.path.join(PROJECT_ROOT, "runs", "ppo")
EVAL_LOG_DIR:   str = os.path.join(PROJECT_ROOT, "runs", "ppo_evals")
BEST_MODEL_SRC: str = os.path.join(MODELS_DIR, "best_model.zip")        # SB3 default
BEST_MODEL_DST: str = os.path.join(MODELS_DIR, "ppo_best.zip")          # we rename post-training
FINAL_MODEL:    str = os.path.join(MODELS_DIR, "ppo_final.zip")

ACTION_NAMES = [a.name for a in Action]    # ["ALLOW","BLOCK","RATE_LIMIT","LOG"]
PROBE_STEPS  = 5_000                       # post-training action-distribution probe


def _print_action_table(actions: np.ndarray, actions_per_class: np.ndarray) -> float:
    """Print overall + per-class action distributions. Returns BLOCK %."""
    total = int(actions.sum())
    print("  Overall action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100.0 * actions[i] / total
        print(f"    {name:<11s} : {int(actions[i]):>6d}  ({pct:5.1f}%)")
    block_pct = 100.0 * actions[Action.BLOCK] / total

    print()
    print("  Per-class action distribution (% per row, ★ = optimal action for class):")
    print(f"    {'class':<8s} {'n':>6s}  "
          f"{'ALLOW':>8s} {'BLOCK':>8s} {'RATE_L':>8s} {'LOG':>8s}")
    for cls in range(5):
        n_cls = int(actions_per_class[cls].sum())
        if n_cls == 0:
            print(f"    {CLASS_NAMES[cls]:<8s} {0:>6d}     —        —        —        —")
            continue
        pcts = 100.0 * actions_per_class[cls] / n_cls
        # Mark optimal: ALLOW for Normal (0); BLOCK for attacks (1..4)
        opt_a = Action.ALLOW if cls == 0 else Action.BLOCK
        marks = ["★" if i == opt_a else " " for i in range(4)]
        print(f"    {CLASS_NAMES[cls]:<8s} {n_cls:>6d}     "
              f"{pcts[0]:5.1f}%{marks[0]}  {pcts[1]:5.1f}%{marks[1]}  "
              f"{pcts[2]:5.1f}%{marks[2]}  {pcts[3]:5.1f}%{marks[3]}")
    return block_pct


def main() -> None:
    t_start = time.time()

    print("=" * 72)
    print(f"PPO Phase 4 — training run")
    print("=" * 72)
    print(f"TOTAL_TIMESTEPS : {TOTAL_TIMESTEPS:,}")
    print(f"N_ENVS          : {N_ENVS}    base_seed={BASE_SEED}    eval_seed={EVAL_SEED}")
    print(f"EVAL_FREQ       : {EVAL_FREQ} vec-env steps  ({EVAL_FREQ * N_ENVS:,} transitions)")
    print(f"N_EVAL_EPISODES : {N_EVAL_EPISODES}/eval")
    print(f"PROC_DIR        : {PROC_DIR}")
    print(f"MODELS_DIR      : {MODELS_DIR}")
    print(f"TB_LOG_DIR      : {TB_LOG_DIR}")
    print(f"EVAL_LOG_DIR    : {EVAL_LOG_DIR}")
    print()
    print("PPO_CONFIG:")
    for k, v in PPO_CONFIG.items():
        print(f"  {k:<16} = {v}")
    print()

    # ── Frozen v1 feature extractor ──────────────────────────────────────────
    ckpt = os.path.join(MODELS_DIR, "lstm_cnn_best.pt")
    print(f"Loading frozen v1 detector: {ckpt}")
    fe = FrozenLSTMCNN(ckpt)
    print(f"  feature_dim={fe.FEATURE_DIM}    (frozen, eval mode, no_grad)")
    print()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading NSL-KDD windows + applying remap_labels (40-class → 5-class)...")
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
    y_train = remap_labels(np.load(os.path.join(PROC_DIR, "y_train.npy")))
    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test  = remap_labels(np.load(os.path.join(PROC_DIR, "y_test.npy")))
    print(f"  X_train {X_train.shape}    y_train {y_train.shape}")
    print(f"  X_test  {X_test.shape}    y_test  {y_test.shape}")
    print(f"  Train class dist: {dict(zip(CLASS_NAMES, [int((y_train==i).sum()) for i in range(5)]))}")
    print(f"  Test  class dist: {dict(zip(CLASS_NAMES, [int((y_test==i).sum()) for i in range(5)]))}")
    print()

    # ── Vec envs ─────────────────────────────────────────────────────────────
    train_env = build_train_env(X_train, y_train, fe, n_envs=N_ENVS, base_seed=BASE_SEED)
    eval_env  = build_eval_env(X_test,  y_test,  fe, seed=EVAL_SEED)
    print(f"train_env: {N_ENVS} envs  (Monitor → FirewallEnv)")
    print(f"eval_env : 1 env   (Monitor → FirewallEnv)")
    print()

    # ── Clear stale TB + eval logs (fresh run each invocation) ───────────────
    for d in (TB_LOG_DIR, EVAL_LOG_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Cleared old logs: {d}")
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)
    # Also clear stale best/final models so we know the artifacts came from THIS run.
    for p in (BEST_MODEL_SRC, BEST_MODEL_DST, FINAL_MODEL):
        if os.path.isfile(p):
            os.remove(p)
            print(f"Cleared old model: {p}")
    print()

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = build_agent(train_env, tb_log_dir=TB_LOG_DIR)
    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"PPO policy params: {n_params:,}")
    print()

    # ── EvalCallback ─────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("━" * 72)
    print(f"agent.learn(total_timesteps={TOTAL_TIMESTEPS:,}, log_interval=1)")
    print("━" * 72)
    t_train = time.time()
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_cb,
        progress_bar=False,
        log_interval=1,
    )
    train_wall = time.time() - t_train
    print("━" * 72)
    print(f"agent.learn() done in {train_wall:.0f}s = {train_wall/60:.2f} min")
    print()

    # ── Persist final + promote best ─────────────────────────────────────────
    agent.save(FINAL_MODEL)
    print(f"Final  model : {FINAL_MODEL}")
    if os.path.isfile(BEST_MODEL_SRC):
        shutil.move(BEST_MODEL_SRC, BEST_MODEL_DST)
        print(f"Best   model : {BEST_MODEL_DST}   (promoted from best_model.zip)")
    else:
        print(f"WARNING: no best_model.zip found — EvalCallback may not have triggered "
              f"a save (this happens if TOTAL_TIMESTEPS < EVAL_FREQ × N_ENVS).")
    print()

    # ── EvalCallback trajectory ──────────────────────────────────────────────
    print("=" * 72)
    print("EvalCallback trajectory (deterministic policy on test set)")
    print("=" * 72)
    print(f"  best_mean_reward : {eval_cb.best_mean_reward:+.4f}")

    eval_npz_path = os.path.join(EVAL_LOG_DIR, "evaluations.npz")
    if os.path.isfile(eval_npz_path):
        ev = np.load(eval_npz_path)
        ts = ev["timesteps"]
        results = ev["results"]              # (n_evals, n_eval_episodes)
        mean_per_eval = results.mean(axis=1)
        std_per_eval  = results.std(axis=1)
        print(f"  n_evals          : {len(ts)}")
        print(f"  reward trajectory:")
        best_idx = int(np.argmax(mean_per_eval))
        for i, (t, m, s) in enumerate(zip(ts, mean_per_eval, std_per_eval)):
            star = " ★" if i == best_idx else ""
            print(f"    eval[{i:>2d}] step={int(t):>7,}   mean={m:+.4f}  std={s:.4f}{star}")
    print()

    # ── Final logger snapshot (train/*) ──────────────────────────────────────
    L = agent.logger.name_to_value
    print("=" * 72)
    print("Final train/* snapshot (from last logger window)")
    print("=" * 72)
    for k in [
        "train/entropy_loss", "train/policy_gradient_loss", "train/value_loss",
        "train/loss",         "train/clip_fraction",        "train/approx_kl",
        "train/learning_rate","train/n_updates",
    ]:
        v = L.get(k)
        if v is not None:
            print(f"  {k:<32} = {v:+.6f}" if isinstance(v, float) else f"  {k:<32} = {v}")
    print()

    # rollout/ep_rew_mean comes from ep_info_buffer — inspect it directly
    buf = agent.ep_info_buffer
    if buf is not None and len(buf) > 0:
        rewards = [e["r"] for e in buf]
        print(f"ep_info_buffer (last {len(buf)} eps):")
        print(f"  mean reward : {np.mean(rewards):+.4f}")
        print(f"  median      : {np.median(rewards):+.4f}")
        print(f"  min / max   : {np.min(rewards):+.4f}  /  {np.max(rewards):+.4f}")
    print()

    # ── Post-training action-distribution probe ──────────────────────────────
    print("=" * 72)
    print(f"Action-distribution probe — best model, deterministic, n={PROBE_STEPS}")
    print("=" * 72)
    best_model_path = BEST_MODEL_DST if os.path.isfile(BEST_MODEL_DST) else FINAL_MODEL
    print(f"Loading model: {best_model_path}")
    best_agent = PPO.load(best_model_path, env=eval_env)

    actions = np.zeros(4, dtype=np.int64)
    actions_per_class = np.zeros((5, 4), dtype=np.int64)
    rewards_probe: list[float] = []

    obs = eval_env.reset()
    for _ in range(PROBE_STEPS):
        action, _ = best_agent.predict(obs, deterministic=True)
        obs, reward, _done, info = eval_env.step(action)
        a = int(action[0])
        true_cls = int(info[0]["true_class"])
        actions[a] += 1
        actions_per_class[true_cls, a] += 1
        rewards_probe.append(float(reward[0]))

    print(f"  probe reward mean : {np.mean(rewards_probe):+.4f}")
    print(f"  probe reward std  : {np.std(rewards_probe):.4f}")
    print(f"  probe reward range: [{np.min(rewards_probe):+.4f}, {np.max(rewards_probe):+.4f}]")
    print()
    block_pct = _print_action_table(actions, actions_per_class)
    print()
    if block_pct > 90:
        print(f"  ⚠️  WARNING: BLOCK rate {block_pct:.1f}% — possible always-BLOCK collapse.")
        print(f"     If per-class table shows BLOCK > 50% on Normal, this is reward hacking.")
    elif block_pct > 70:
        print(f"  ⚠️  BLOCK rate {block_pct:.1f}% high — check per-class breakdown above for Normal column.")
    else:
        print(f"  ✅ BLOCK rate {block_pct:.1f}% — no always-BLOCK collapse.")
    print()

    # ── Wall summary ─────────────────────────────────────────────────────────
    total_wall = time.time() - t_start
    print("=" * 72)
    print(f"Phase 4 PPO training complete.")
    print("=" * 72)
    print(f"  train wall      : {train_wall:.0f}s = {train_wall/60:.2f} min")
    print(f"  total wall      : {total_wall:.0f}s = {total_wall/60:.2f} min  (includes setup + probe)")
    print(f"  best mean reward: {eval_cb.best_mean_reward:+.4f}")
    print(f"  best model      : {BEST_MODEL_DST}")
    print(f"  final model     : {FINAL_MODEL}")
    print(f"  TB              : tensorboard --logdir {TB_LOG_DIR}")
    print(f"  Eval log        : {EVAL_LOG_DIR}")


if __name__ == "__main__":
    main()
