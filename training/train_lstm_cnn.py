"""
Phase 3 — LSTM-CNN training script for the rl-firewall project.

Usage (from project root, venv activated):
    python training/train_lstm_cnn.py

What this script does
---------------------
1. Loads preprocessed NSL-KDD windows from data/processed/
2. Remaps the 40-class fine-grained LabelEncoder integers → 5 coarse classes
   (Normal / DoS / Probe / R2L / U2R)
3. Builds PyTorch DataLoaders with class-balanced loss weights
4. Trains the LSTMCNN model for up to 30 epochs with early stopping (patience=5)
5. Logs metrics to TensorBoard (runs/lstm_cnn/)
6. Saves the best checkpoint to data/models/lstm_cnn_best.pt
7. Produces a full classification report, confusion matrix, ROC curves, and
   training curves — all saved as PNGs to data/processed/
"""

# ── Section 1: sys.path injection ─────────────────────────────────────────────
# Must happen before any project-local imports so Python can find config.py
# and the src/ package regardless of the working directory.
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../training/
PROJECT_ROOT = os.path.dirname(_HERE)                 # .../rl-firewall/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Section 2: Imports ─────────────────────────────────────────────────────────
import numpy as np

import matplotlib                        # non-interactive backend must be set
matplotlib.use("Agg")                   # BEFORE importing pyplot
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import PROC_DIR, MODELS_DIR
from src.models.lstm_cnn import LSTMCNN
from src.preprocessing.labels import remap_labels

# ── Section 3: Constants ───────────────────────────────────────────────────────
BATCH_SIZE       = 256
N_EPOCHS         = 50
LR               = 3e-4
WARMUP_EPOCHS    = 3            # linear warmup: lr scaled by epoch/WARMUP_EPOCHS
PATIENCE         = 5            # early-stopping patience (epochs without val_loss improvement)
SCHED_PAT        = 3            # ReduceLROnPlateau patience
SCHED_FACTOR     = 0.5
GRAD_CLIP_NORM   = 1.0          # max gradient norm (gradient clipping)
MAX_CLASS_WEIGHT = 50.0         # cap on per-class loss weights to prevent gradient blow-up
OVERSAMPLE_TO    = 1000         # rare-class oversampling target (train set only)

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# LABEL_REMAP and remap_labels are imported from src.preprocessing.labels —
# single source of truth for both Phase 3 (this script) and Phase 4 (PPO env).
# See §12.1 of the project documentation for the discovery story.


# ── Section 4: Helpers ─────────────────────────────────────────────────────────


def oversample_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    min_per_class: int = OVERSAMPLE_TO,
    seed: int = 42,
) -> tuple:
    """Duplicate samples of any class with fewer than `min_per_class` examples.

    Sampling is with replacement on the existing minority-class indices.
    The returned arrays are shuffled so duplicates are interleaved with the
    rest of the data (avoids batch composition artefacts during training).
    """
    rng = np.random.default_rng(seed)
    extra_X_chunks: list = []
    extra_y_chunks: list = []

    for cls in np.unique(y):
        n_cls = int((y == cls).sum())
        if n_cls < min_per_class:
            n_extra = min_per_class - n_cls
            cls_idx = np.where(y == cls)[0]
            picked  = rng.choice(cls_idx, size=n_extra, replace=True)
            extra_X_chunks.append(X[picked])
            extra_y_chunks.append(y[picked])

    if not extra_X_chunks:
        return X, y

    X_aug = np.concatenate([X, *extra_X_chunks], axis=0)
    y_aug = np.concatenate([y, *extra_y_chunks], axis=0)

    # Shuffle so the duplicated rare-class rows aren't all clumped at the end.
    perm = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays in a DataLoader."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,    # required on Windows — multiprocessing spawn causes errors
        pin_memory=False, # no CUDA on this machine
    )


def save_training_curves(
    train_losses: list,
    val_losses: list,
    val_accs: list,
    val_f1s: list,
    out_dir: str,
) -> None:
    """Plot and save loss + accuracy/F1 curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(epochs, val_accs, label="Val Accuracy")
    axes[1].plot(epochs, val_f1s,  label="Val F1-macro")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Accuracy & F1-macro")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, "lstm_cnn_training_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {path}")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
) -> None:
    """Plot and save confusion matrix heatmap (no seaborn dependency)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — LSTM-CNN on NSL-KDD Test Set")
    plt.tight_layout()

    path = os.path.join(out_dir, "lstm_cnn_confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def save_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    out_dir: str,
) -> None:
    """Plot per-class ROC curves and print macro OVR AUC."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])  # (N, 5)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(CLASS_NAMES):
        if y_bin[:, i].sum() == 0:
            print(f"  Skipping ROC for {name} — no positive samples in test set.")
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest (LSTM-CNN, NSL-KDD)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(out_dir, "lstm_cnn_roc_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curves      → {path}")

    try:
        macro_auc = roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro"
        )
        print(f"\nROC-AUC (macro OVR): {macro_auc:.4f}")
    except ValueError as exc:
        print(f"ROC-AUC could not be computed: {exc}")


# ── Section 5: Main ────────────────────────────────────────────────────────────

def main() -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"PROC_DIR    : {PROC_DIR}")
    print(f"MODELS_DIR  : {MODELS_DIR}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Data loading & label remapping ────────────────────────────────────────
    print("\nLoading data...")
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))   # (N, 10, 41)
    y_train = remap_labels(np.load(os.path.join(PROC_DIR, "y_train.npy")))

    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test  = remap_labels(np.load(os.path.join(PROC_DIR, "y_test.npy")))

    print(f"  X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    # Sanity: verify remapping produced only classes 0–4
    assert set(np.unique(y_train)).issubset({0, 1, 2, 3, 4}), "Unexpected train labels after remap"
    assert set(np.unique(y_test)).issubset({0, 1, 2, 3, 4}),  "Unexpected test labels after remap"

    print("\n  Train class distribution (raw):")
    for cls, name in enumerate(CLASS_NAMES):
        count = (y_train == cls).sum()
        print(f"    {name:<10} {count:>7,}")
    print("\n  Test class distribution:")
    for cls, name in enumerate(CLASS_NAMES):
        count = (y_test == cls).sum()
        print(f"    {name:<10} {count:>7,}")

    # ── Oversample rare training classes (R2L, U2R) ───────────────────────────
    # Test set is left untouched — oversampling is a training-only intervention.
    print(f"\nOversampling training classes with < {OVERSAMPLE_TO:,} samples...")
    X_train, y_train = oversample_rare_classes(X_train, y_train, OVERSAMPLE_TO)
    print(f"  X_train (after oversample): {X_train.shape}")
    print("  Train class distribution (oversampled):")
    for cls, name in enumerate(CLASS_NAMES):
        count = (y_train == cls).sum()
        print(f"    {name:<10} {count:>7,}")

    # ── Class weights (computed on the oversampled train set) ─────────────────
    cw = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2, 3, 4]),
        y=y_train,
    )
    cw_raw = cw.copy()
    cw = np.clip(cw, 0.0, MAX_CLASS_WEIGHT)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"\nClass weights (raw)    : {dict(zip(CLASS_NAMES, cw_raw.round(2)))}")
    print(f"Class weights (capped) : {dict(zip(CLASS_NAMES, cw.round(2)))}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # ── Model, optimiser, loss, scheduler ─────────────────────────────────────
    model     = LSTMCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHED_PAT, factor=SCHED_FACTOR
    )

    tb_log_dir = os.path.join(PROJECT_ROOT, "runs", "lstm_cnn")
    if os.path.isdir(tb_log_dir):
        import shutil
        shutil.rmtree(tb_log_dir)
        print(f"\nCleared old TensorBoard logs at {tb_log_dir}")
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log dir : {tb_log_dir}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters    : {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    best_model_path  = os.path.join(MODELS_DIR, "lstm_cnn_best.pt")

    train_losses: list = []
    val_losses:   list = []
    val_accs:     list = []
    val_f1s:      list = []

    print(f"\n{'─'*75}")
    print(f"{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>10}  {'val_acc':>8}  {'val_f1':>8}")
    print(f"{'─'*75}")

    for epoch in range(1, N_EPOCHS + 1):
        # ── Linear LR warmup for the first WARMUP_EPOCHS epochs ───────────────
        # epoch 1 → LR/3, epoch 2 → 2*LR/3, epoch 3 → LR.  After warmup the
        # ReduceLROnPlateau scheduler takes over via scheduler.step(val_loss).
        if epoch <= WARMUP_EPOCHS:
            warm_lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ── Evaluate on test set ───────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        all_preds:  list = []
        all_labels: list = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss_sum += criterion(logits, yb).item() * Xb.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss = val_loss_sum / len(test_loader.dataset)
        val_acc  = float((np.array(all_preds) == np.array(all_labels)).mean())
        val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # Scheduler steps on validation loss — only after warmup is done so it
        # doesn't penalise the artificially-low warmup learning rate.
        if epoch > WARMUP_EPOCHS:
            scheduler.step(val_loss)

        # TensorBoard
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("F1/val",     val_f1,     epoch)
        writer.add_scalar("LR",         current_lr, epoch)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            marker = "  ← best"
        else:
            patience_counter += 1

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}"
            f"  {val_acc:>8.4f}  {val_f1:>8.4f}{marker}"
        )

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch} (patience={PATIENCE}).")
            break

    writer.close()
    print(f"{'─'*75}")
    print(f"\nBest val_loss : {best_val_loss:.4f}")
    print(f"Best model    : {best_model_path}")

    # ── Post-training evaluation ───────────────────────────────────────────────
    print("\n" + "═"*75)
    print("POST-TRAINING EVALUATION (test set, best checkpoint)")
    print("═"*75)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    all_preds_np: list = []
    all_labels_np: list = []
    all_probs: list = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs  = torch.softmax(logits, dim=1)
            all_preds_np.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels_np.extend(yb.numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred  = np.array(all_preds_np)
    y_true  = np.array(all_labels_np)
    y_probs = np.vstack(all_probs)         # (N_test, 5)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    # Plots
    print("Saving plots...")
    save_training_curves(train_losses, val_losses, val_accs, val_f1s, PROC_DIR)
    save_confusion_matrix(y_true, y_pred, PROC_DIR)
    save_roc_curves(y_true, y_probs, PROC_DIR)

    print("\nPhase 3 complete.")
    print(f"  TensorBoard : tensorboard --logdir {os.path.join(PROJECT_ROOT, 'runs', 'lstm_cnn')}")


if __name__ == "__main__":
    main()
