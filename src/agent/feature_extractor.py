"""
Phase 4 — frozen v1 LSTM-CNN feature extractor for the PPO agent.

Wraps ``src.models.lstm_cnn.LSTMCNN`` and exposes the 256-dim BiLSTM output
(last timestep, before the 5-class classifier head) as the PPO observation.
The classifier head is loaded into the wrapped module — so the v1 checkpoint
loads with ``strict=True`` — but never invoked during forward, so no
classifier gradients exist for PPO to interfere with.

The wrapper is frozen on three layers:

1. ``model.eval()`` puts dropout in inference mode and BatchNorm on its
   stored running statistics (no train-mode contamination from PPO inputs).
2. ``requires_grad_(False)`` on every parameter — gradients can't flow back
   into the detector even if SB3 mistakenly enables them.
3. ``@torch.no_grad()`` on ``forward()`` — guarantees no autograd graph is
   built during a PPO rollout, freeing memory and CPU.

This three-layer freeze matches §9.6 / §20 of the project documentation
("frozen detector through Phase 5"). The §21 v2 attempts left the v1
checkpoint at ``data/models/lstm_cnn_best.pt`` as the
canonical detector for Phase 4.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from src.models.lstm_cnn import LSTMCNN


class FrozenLSTMCNN(nn.Module):
    """Frozen v1 LSTM-CNN, exposing the 256-dim BiLSTM output as PPO state.

    Parameters
    ----------
    ckpt_path : str
        Absolute path to ``lstm_cnn_best.pt`` (raw state_dict — no wrapper
        keys like ``"model_state_dict"``; matches what Phase 3 saved).
    device : str
        ``"cpu"`` or ``"cuda"``. Default ``"cpu"`` — no CUDA on this machine.

    Attributes
    ----------
    FEATURE_DIM : int
        Output dimensionality of ``forward()``. Matches lstm_hidden (128) × 2
        for the bidirectional LSTM. Exposed as a class attribute so
        ``FirewallEnv`` can build its observation space without instantiating
        the model first.
    T_WINDOW : int
        Required sliding-window length on the input (10).
    N_FEATURES : int
        Required per-timestep feature count on the input (41).
    """

    FEATURE_DIM: int = 256
    T_WINDOW:    int = 10
    N_FEATURES:  int = 41

    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        super().__init__()
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        self._device = torch.device(device)

        # Build the same architecture the checkpoint was trained with and
        # load weights with strict=True so any future architecture drift
        # raises immediately rather than silently zero-initialising.
        self._inner = LSTMCNN()
        state = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        self._inner.load_state_dict(state, strict=True)

        # Freeze layer 1: eval mode (dropout off, BatchNorm uses running stats).
        self._inner.eval()

        # Freeze layer 2: requires_grad=False on every parameter.
        for p in self._inner.parameters():
            p.requires_grad_(False)

        self._inner.to(self._device)

    @torch.no_grad()  # Freeze layer 3: no autograd graph during rollout.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a batch of windows into the 256-dim PPO state space.

        Replicates ``LSTMCNN.forward`` up to (but not including) the
        classifier head. The reference is the line ``x = out[:, -1, :]``
        in ``src/models/lstm_cnn.py`` — that is the tensor we return.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T_WINDOW=10, N_FEATURES=41)``, dtype float32.

        Returns
        -------
        torch.Tensor
            Shape ``(B, FEATURE_DIM=256)``, dtype float32, ``requires_grad=False``,
            no ``grad_fn``.
        """
        x = x.to(self._device)
        x = x.permute(0, 2, 1)              # (B, 41, 10)  — Conv1d expects (N, C, L)
        x = self._inner.cnn(x)              # (B, 128, 5)  — MaxPool halves seq_len
        x = x.permute(0, 2, 1)              # (B, 5, 128)  — LSTM expects (N, L, C)
        out, _ = self._inner.lstm(x)        # (B, 5, 256)  — bidirectional doubles hidden
        return out[:, -1, :]                # (B, 256)     — last timestep, PPO state

    def extra_repr(self) -> str:
        return f"frozen=True, device={self._device}, feature_dim={self.FEATURE_DIM}"
