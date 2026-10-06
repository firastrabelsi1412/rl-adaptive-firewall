"""
Phase 4 — Step 3e — PPO firewall agent evaluation vs 3 baselines.

Loads ``data/models/ppo_best.zip`` (selected by EvalCallback
during the §22.7 training run — eval[6] @ 140k steps, mean reward −0.1063)
and evaluates it deterministically on the full NSL-KDD test set (22,534
windows after Phase 2 sliding-window construction), comparing against:

  B1 — Random uniform policy (lower bound; analytical reward ≈ −0.58 per §22.2.C)
  B2 — Always-BLOCK    (tests whether PPO discovers anything beyond "block everything")
  B3 — LSTM-CNN argmax (the "Baseline 2" of §18 — detection without RL,
        action = ALLOW if predicted_class == Normal else BLOCK)

Detection rate / FPR follow firewall semantics — an action is "blocking"
iff it equals BLOCK. RATE_LIMIT and LOG are *informational* responses; per
§22.2.B they don't actually drop the packet. For PPO we additionally report
a "broad detection" rate (any non-ALLOW) as context, since the trained
policy chose LOG on R2L/U2R (see §12.5 detector-shift discussion).

The four PNG outputs land in ``data/processed/`` per §22.7:

  ppo_reward_curve.png        — 25 eval checkpoints over the 500k training run
  ppo_action_distribution.png — overall action mix, PPO vs each baseline
  ppo_per_class_actions.png   — stacked-bar per-class action distribution (PPO)
  ppo_vs_baselines.png        — mean reward, detection rate, FPR side-by-side

Per §22.8: the reward function here uses **ground-truth labels** (the
"oracle reward"). The deployment story replaces this with production FPR
feedback in §16.3 Loop 1 — summer extension work, out of scope for Phase 4.

Usage
-----
    python evaluation/evaluate_ppo.py 2>&1 | tee data/processed/_ppo_eval_log.txt
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
import json
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from config import PROC_DIR, MODELS_DIR
from src.agent.feature_extractor import FrozenLSTMCNN
from src.environment.firewall_env import (
    Action,
    CLASS_NAMES,
    LATENCY_PENALTY,
    REWARD_TABLE,
)
from src.preprocessing.labels import remap_labels


# ── Constants ─────────────────────────────────────────────────────────────────

ACTION_NAMES = [a.name for a in Action]   # ["ALLOW","BLOCK","RATE_LIMIT","LOG"]
N_ACTIONS   = len(Action)                 # 4
N_CLASSES   = len(CLASS_NAMES)            # 5

BATCH_SIZE = 1024                          # for the frozen-extractor forward pass

PPO_MODEL_PATH    = os.path.join(MODELS_DIR, "ppo_best.zip")
DETECTOR_CKPT     = os.path.join(MODELS_DIR, "lstm_cnn_best.pt")
EVALS_NPZ_PATH    = os.path.join(PROJECT_ROOT, "runs", "ppo_evals", "evaluations.npz")

REWARD_CURVE_PNG  = os.path.join(PROC_DIR, "ppo_reward_curve.png")
ACTION_DIST_PNG   = os.path.join(PROC_DIR, "ppo_action_distribution.png")
PER_CLASS_PNG     = os.path.join(PROC_DIR, "ppo_per_class_actions.png")
VS_BASELINES_PNG  = os.path.join(PROC_DIR, "ppo_vs_baselines.png")
RESULTS_JSON      = os.path.join(PROC_DIR, "ppo_eval_results.json")

# Reproducibility for B1 random baseline.
B1_SEED = 1234

# Action codes (for readability)
A_ALLOW, A_BLOCK, A_RATE, A_LOG = int(Action.ALLOW), int(Action.BLOCK), int(Action.RATE_LIMIT), int(Action.LOG)


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_states(X: np.ndarray, extractor: FrozenLSTMCNN) -> np.ndarray:
    """Batched forward through the frozen v1 BiLSTM up to the 256-dim state.

    Returns shape ``(N, 256)`` float32. Single full-test-set pass is fine on
    CPU — Phase 4 design has been verified at 278 steps/s; 22.5k windows takes
    a couple of minutes at most.
    """
    N = X.shape[0]
    out = np.empty((N, extractor.FEATURE_DIM), dtype=np.float32)
    for i in range(0, N, BATCH_SIZE):
        batch = torch.from_numpy(X[i:i + BATCH_SIZE]).to(torch.float32)
        states = extractor(batch).cpu().numpy().astype(np.float32, copy=False)
        out[i:i + BATCH_SIZE] = states
    return out


def lstm_cnn_argmax_actions(states: np.ndarray, extractor: FrozenLSTMCNN) -> tuple[np.ndarray, np.ndarray]:
    """B3 baseline: run the v1 classifier head on the 256-dim states.

    The classifier head ``Linear(256→128) → ReLU → Dropout → Linear(128→5)``
    is loaded into the frozen wrapper but never invoked in the env path.
    We invoke it here to get the §18 "Baseline 2" predictions, then map:

        predicted_class == Normal (0)  ->  ALLOW (0)
        predicted_class != Normal      ->  BLOCK (1)

    Returns
    -------
    (actions, predicted_classes) : both np.ndarray of shape (N,), dtype int64
    """
    N = states.shape[0]
    predicted = np.empty(N, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            chunk = torch.from_numpy(states[i:i + BATCH_SIZE]).to(torch.float32)
            logits = extractor._inner.classifier(chunk)   # (B, 5)
            predicted[i:i + BATCH_SIZE] = logits.argmax(dim=1).cpu().numpy()
    actions = np.where(predicted == 0, A_ALLOW, A_BLOCK).astype(np.int64)
    return actions, predicted


def ppo_actions(model: PPO, states: np.ndarray) -> np.ndarray:
    """Deterministic argmax actions from the PPO policy on a batch of states."""
    N = states.shape[0]
    actions = np.empty(N, dtype=np.int64)
    # SB3 predict accepts batched obs; do it in one shot for speed.
    # (Internally it builds a tensor on the policy's device.)
    pred, _ = model.predict(states, deterministic=True)
    actions[:] = pred.astype(np.int64).reshape(-1)
    assert actions.shape == (N,)
    return actions


def compute_rewards(actions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised reward lookup: REWARD_TABLE[action, true_class] - LATENCY_PENALTY."""
    return REWARD_TABLE[actions, y].astype(np.float32) - LATENCY_PENALTY


def action_distribution(actions: np.ndarray) -> np.ndarray:
    """Return (4,) int counts for ALLOW/BLOCK/RATE_LIMIT/LOG."""
    return np.bincount(actions, minlength=N_ACTIONS).astype(np.int64)


def per_class_action_distribution(actions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return (5, 4) int counts: rows = true class, cols = action."""
    table = np.zeros((N_CLASSES, N_ACTIONS), dtype=np.int64)
    for cls in range(N_CLASSES):
        mask = (y == cls)
        if mask.any():
            table[cls] = np.bincount(actions[mask], minlength=N_ACTIONS)
    return table


def detection_metrics(actions: np.ndarray, y: np.ndarray) -> dict:
    """Detection rate and FPR — strict ('BLOCK only') and broad ('any non-ALLOW').

    DR (strict) = P(action == BLOCK | true_class != Normal)
    FPR (strict) = P(action == BLOCK | true_class == Normal)
    DR (broad)  = P(action != ALLOW | true_class != Normal)
    FPR (broad) = P(action != ALLOW | true_class == Normal)
    """
    is_attack  = (y != 0)
    is_normal  = (y == 0)
    is_block   = (actions == A_BLOCK)
    is_nonallow = (actions != A_ALLOW)

    n_attack = int(is_attack.sum())
    n_normal = int(is_normal.sum())

    dr_strict  = float((is_block & is_attack).sum()) / max(n_attack, 1)
    fpr_strict = float((is_block & is_normal).sum()) / max(n_normal, 1)
    dr_broad   = float((is_nonallow & is_attack).sum()) / max(n_attack, 1)
    fpr_broad  = float((is_nonallow & is_normal).sum()) / max(n_normal, 1)
    return {
        "n_attack": n_attack,
        "n_normal": n_normal,
        "dr_strict":  dr_strict,
        "fpr_strict": fpr_strict,
        "dr_broad":   dr_broad,
        "fpr_broad":  fpr_broad,
    }


# ── Pretty-printers ───────────────────────────────────────────────────────────

def print_action_table(name: str, actions: np.ndarray, y: np.ndarray) -> None:
    print(f"── {name} ──────────────────────────────────────────────────────────")
    overall = action_distribution(actions)
    per_cls = per_class_action_distribution(actions, y)
    total = int(overall.sum())

    # Overall row (counts + pct)
    print("  Overall action distribution:")
    for i, an in enumerate(ACTION_NAMES):
        pct = 100.0 * overall[i] / total if total else 0.0
        print(f"    {an:<11s} : {int(overall[i]):>6d}  ({pct:5.1f}%)")

    # Per-class breakdown — counts AND percentages (U2R is n=20 in test set,
    # so percentages on their own are noisy)
    print()
    print("  Per-class action distribution  (count [pct]   ★ = optimal action)")
    print(f"    {'class':<8s} {'n':>6s}   "
          f"{'ALLOW':>14s} {'BLOCK':>14s} {'RATE_LIMIT':>14s} {'LOG':>14s}")
    for cls in range(N_CLASSES):
        n_cls = int(per_cls[cls].sum())
        if n_cls == 0:
            print(f"    {CLASS_NAMES[cls]:<8s} {0:>6d}     "
                  f"{'—':>14s} {'—':>14s} {'—':>14s} {'—':>14s}")
            continue
        opt_a = A_ALLOW if cls == 0 else A_BLOCK  # Normal: ALLOW (tied w/ LOG); attacks: BLOCK
        cells = []
        for a in range(N_ACTIONS):
            cnt = int(per_cls[cls, a])
            pct = 100.0 * cnt / n_cls
            star = "★" if a == opt_a else " "
            cells.append(f"{cnt:>5d} [{pct:4.1f}%]{star}")
        print(f"    {CLASS_NAMES[cls]:<8s} {n_cls:>6d}     "
              f"{cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s} {cells[3]:>14s}")

    # Detection metrics
    dm = detection_metrics(actions, y)
    rewards = compute_rewards(actions, y)
    print()
    print(f"  mean reward      : {rewards.mean():+.4f}   (std {rewards.std():.4f})")
    print(f"  reward range     : [{rewards.min():+.4f}, {rewards.max():+.4f}]")
    print(f"  detection rate   : {dm['dr_strict']:.4f}  (BLOCK only,  n_attack={dm['n_attack']})")
    print(f"     broad         : {dm['dr_broad']:.4f}  (BLOCK|RATE|LOG)")
    print(f"  false-pos rate   : {dm['fpr_strict']:.4f}  (BLOCK only,  n_normal={dm['n_normal']})")
    print(f"     broad         : {dm['fpr_broad']:.4f}  (BLOCK|RATE|LOG)")
    print()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_reward_curve(out_path: str) -> bool:
    """Plot the EvalCallback reward trajectory across the 500k-step training run.

    Returns True if the npz was found and plotted; False otherwise.
    """
    if not os.path.isfile(EVALS_NPZ_PATH):
        print(f"[plot] evaluations.npz not found at {EVALS_NPZ_PATH} — skipping reward curve.")
        return False
    ev = np.load(EVALS_NPZ_PATH)
    ts      = ev["timesteps"]                       # (n_evals,)
    results = ev["results"]                          # (n_evals, n_eval_episodes)
    means = results.mean(axis=1)
    stds  = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, means, marker="o", linewidth=1.5, color="#1f77b4", label="eval mean reward")
    ax.fill_between(ts, means - stds, means + stds, alpha=0.15, color="#1f77b4", label="±1σ across 1k eval eps")
    # Reference lines
    ax.axhline(0.0,    color="#444", linestyle="--", linewidth=0.8, label="zero")
    ax.axhline(+0.07,  color="#2ca02c", linestyle=":", linewidth=0.8, label="oracle ceiling +0.07")
    ax.axhline(-0.36,  color="#ff7f0e", linestyle=":", linewidth=0.8, label="always-BLOCK ≈ −0.36")
    ax.axhline(-0.58,  color="#d62728", linestyle=":", linewidth=0.8, label="random ≈ −0.58")
    best_idx = int(np.argmax(means))
    ax.scatter([ts[best_idx]], [means[best_idx]], s=120, facecolors="none", edgecolors="#1f77b4",
               linewidths=2, label=f"best ★ step={int(ts[best_idx]):,}, r={means[best_idx]:+.4f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean eval reward (n=1000 deterministic eps)")
    ax.set_title("PPO eval reward over 500k training steps (Phase 4 Step 3d)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    return True


def plot_action_distribution(by_method: dict[str, np.ndarray], out_path: str) -> None:
    """Side-by-side bar chart of overall action distributions for PPO + B1/B2/B3."""
    methods = list(by_method.keys())
    n_methods = len(methods)
    x = np.arange(N_ACTIONS)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, m in enumerate(methods):
        counts = by_method[m]
        total = max(counts.sum(), 1)
        pct = 100.0 * counts / total
        bars = ax.bar(x + i * width - 0.4 + width / 2, pct, width, label=m, color=colors[i % len(colors)])
        for b, p in zip(bars, pct):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{p:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_ylabel("% of test windows")
    ax.set_title("Overall action distribution — PPO vs baselines (full test set)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_per_class_actions(per_cls: np.ndarray, out_path: str) -> None:
    """Stacked bar chart of per-class action distribution for the PPO policy.

    per_cls shape: (5, 4) — rows = true class, cols = action count
    """
    pct = per_cls / np.maximum(per_cls.sum(axis=1, keepdims=True), 1) * 100.0

    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(N_CLASSES, dtype=np.float64)
    action_colors = {
        A_ALLOW: "#2ca02c",
        A_BLOCK: "#d62728",
        A_RATE:  "#ff7f0e",
        A_LOG:   "#7f7f7f",
    }
    for a in range(N_ACTIONS):
        ax.bar(CLASS_NAMES, pct[:, a], bottom=bottoms,
               label=ACTION_NAMES[a], color=action_colors[a], edgecolor="white", linewidth=0.5)
        # annotate slices that are >= 5%
        for cls in range(N_CLASSES):
            if pct[cls, a] >= 5.0:
                ax.text(cls, bottoms[cls] + pct[cls, a] / 2,
                        f"{pct[cls, a]:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += pct[:, a]

    # annotate n above each bar
    n_per_class = per_cls.sum(axis=1)
    for cls, n in enumerate(n_per_class):
        ax.text(cls, 101, f"n={int(n)}", ha="center", va="bottom", fontsize=8, color="black")

    ax.set_ylim(0, 110)
    ax.set_ylabel("% of windows in class")
    ax.set_title("PPO per-class action distribution (deterministic, full test set)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_vs_baselines(summary: dict, out_path: str) -> None:
    """Three subplots: mean reward, detection rate (strict), FPR (strict)."""
    methods = list(summary.keys())
    rewards = [summary[m]["mean_reward"] for m in methods]
    drs     = [summary[m]["dr_strict"]   for m in methods]
    fprs    = [summary[m]["fpr_strict"]  for m in methods]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(methods)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    ax = axes[0]
    bars = ax.bar(methods, rewards, color=colors)
    ax.axhline(0.0,    color="#444", linestyle="--", linewidth=0.7)
    ax.axhline(+0.07,  color="#2ca02c", linestyle=":", linewidth=0.7, label="oracle +0.07")
    ax.axhline(-0.58,  color="#d62728", linestyle=":", linewidth=0.7, label="random −0.58")
    for b, r in zip(bars, rewards):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{r:+.3f}", ha="center",
                va="bottom" if r >= 0 else "top", fontsize=9)
    ax.set_title("Mean reward (full test)")
    ax.set_ylabel("mean reward / step")
    ax.set_ylim(-1.0, 0.6)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(methods, drs, color=colors)
    for b, d in zip(bars, drs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{d:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Detection rate  (action == BLOCK | attack)")
    ax.set_ylabel("DR")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    bars = ax.bar(methods, fprs, color=colors)
    for b, f in zip(bars, fprs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("False-positive rate  (action == BLOCK | normal)")
    ax.set_ylabel("FPR")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("PPO vs baselines — Phase 4 Step 3e headline metrics", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("=" * 72)
    print("PPO Phase 4 — Step 3e — evaluation vs 3 baselines")
    print("=" * 72)
    print(f"  PPO model     : {PPO_MODEL_PATH}")
    print(f"  Detector ckpt : {DETECTOR_CKPT}   (frozen v1, §21 confirmed)")
    print(f"  PROC_DIR      : {PROC_DIR}")
    print()

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading X_test, y_test and applying remap_labels (40-class → 5-class)...")
    X_test = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test = remap_labels(np.load(os.path.join(PROC_DIR, "y_test.npy")))
    print(f"  X_test  {X_test.shape}    y_test  {y_test.shape}")
    class_counts = {CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(N_CLASSES)}
    print(f"  Class distribution: {class_counts}")
    print()

    # ── Load frozen detector ─────────────────────────────────────────────────
    print(f"Loading frozen v1 detector...")
    fe = FrozenLSTMCNN(DETECTOR_CKPT)
    print(f"  feature_dim={fe.FEATURE_DIM}    (frozen, eval mode, no_grad)")
    print()

    # ── Encode all windows once ──────────────────────────────────────────────
    print(f"Encoding {len(X_test):,} test windows through frozen BiLSTM...")
    t_enc = time.time()
    states = encode_states(X_test, fe)
    print(f"  states {states.shape}    encode wall {time.time()-t_enc:.1f}s")
    assert states.shape == (len(X_test), fe.FEATURE_DIM)
    assert states.dtype == np.float32
    print()

    # ── Load PPO agent ───────────────────────────────────────────────────────
    print(f"Loading PPO agent...")
    if not os.path.isfile(PPO_MODEL_PATH):
        raise FileNotFoundError(f"PPO model not found: {PPO_MODEL_PATH}")
    # No env needed for predict-only (we feed obs directly).
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")
    print(f"  policy params: {sum(p.numel() for p in ppo.policy.parameters()):,}")
    print()

    # ── Compute actions for each method ──────────────────────────────────────
    print("Computing actions for PPO and B1/B2/B3 baselines...")
    t_act = time.time()
    a_ppo = ppo_actions(ppo, states)

    rng = np.random.default_rng(B1_SEED)
    a_b1 = rng.integers(0, N_ACTIONS, size=len(y_test)).astype(np.int64)

    a_b2 = np.full(len(y_test), A_BLOCK, dtype=np.int64)

    a_b3, predicted_b3 = lstm_cnn_argmax_actions(states, fe)

    print(f"  actions computed in {time.time()-t_act:.1f}s")
    print()

    # ── Per-method reports ───────────────────────────────────────────────────
    methods = {
        "PPO (best)":         a_ppo,
        "B1 random":          a_b1,
        "B2 always-BLOCK":    a_b2,
        "B3 LSTM-CNN argmax": a_b3,
    }
    summary: dict[str, dict] = {}
    for name, acts in methods.items():
        rewards = compute_rewards(acts, y_test)
        dm = detection_metrics(acts, y_test)
        overall = action_distribution(acts)
        per_cls = per_class_action_distribution(acts, y_test)
        summary[name] = {
            "mean_reward":   float(rewards.mean()),
            "std_reward":    float(rewards.std()),
            "min_reward":    float(rewards.min()),
            "max_reward":    float(rewards.max()),
            "dr_strict":     dm["dr_strict"],
            "fpr_strict":    dm["fpr_strict"],
            "dr_broad":      dm["dr_broad"],
            "fpr_broad":     dm["fpr_broad"],
            "n_attack":      dm["n_attack"],
            "n_normal":      dm["n_normal"],
            "overall_counts": overall.tolist(),
            "per_class_counts": per_cls.tolist(),
        }
        print_action_table(name, acts, y_test)

    # ── Headline table — PPO vs B3 (the §22.9 deliverable) ───────────────────
    print("=" * 72)
    print("HEADLINE — PPO vs B3 LSTM-CNN argmax  (the Phase 4 final-report tuple)")
    print("=" * 72)
    print(f"  {'metric':<22s} {'PPO':>12s} {'B3 LSTM-CNN':>14s} {'Δ (PPO−B3)':>14s}")
    for label, key in [
        ("mean reward",    "mean_reward"),
        ("detection rate", "dr_strict"),
        ("FPR",            "fpr_strict"),
        ("DR (broad)",     "dr_broad"),
        ("FPR (broad)",    "fpr_broad"),
    ]:
        v_ppo = summary["PPO (best)"][key]
        v_b3  = summary["B3 LSTM-CNN argmax"][key]
        delta = v_ppo - v_b3
        sgn = "+" if delta >= 0 else ""
        print(f"  {label:<22s} {v_ppo:>+12.4f} {v_b3:>+14.4f} {sgn}{delta:>+13.4f}")
    print()

    # ── §22.8 acknowledgement ────────────────────────────────────────────────
    print("Note — oracle reward function:")
    print("  Phase 4 PPO is trained and evaluated against an *oracle* reward derived")
    print("  from ground-truth labels (the 4×5 grid in §22.2.B). Production deployment")
    print("  replaces this with §16.3 Loop 1 — refined reward from measured production")
    print("  FPR — which is summer-extension work and out of scope for Phase 4.")
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    print("Generating plots...")
    plot_reward_curve(REWARD_CURVE_PNG)
    plot_action_distribution(
        {name: np.array(summary[name]["overall_counts"]) for name in methods},
        ACTION_DIST_PNG,
    )
    plot_per_class_actions(np.array(summary["PPO (best)"]["per_class_counts"]), PER_CLASS_PNG)
    plot_vs_baselines(summary, VS_BASELINES_PNG)
    print()

    # ── Persist machine-readable summary ─────────────────────────────────────
    out = {
        "ppo_model":      PPO_MODEL_PATH,
        "detector_ckpt":  DETECTOR_CKPT,
        "n_test_windows": int(len(y_test)),
        "class_counts":   class_counts,
        "action_names":   ACTION_NAMES,
        "class_names":    CLASS_NAMES,
        "latency_penalty": LATENCY_PENALTY,
        "methods":        summary,
        "b1_seed":        B1_SEED,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[json] saved {RESULTS_JSON}")
    print()

    print("=" * 72)
    print(f"Phase 4 Step 3e evaluation complete in {time.time()-t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
