"""
LSTM-CNN hybrid model for network intrusion detection.

Architecture overview
---------------------
Input: (batch, T=10, F=41)  — sliding window of T timesteps, F features each.

1. CNN block  — extracts local temporal patterns across the feature dimension.
   Input is permuted to (batch, F, T) so Conv1d treats features as channels.
   Three Conv1d layers (64 → 128 → 128) with BatchNorm + ReLU, then MaxPool1d(2)
   halves the sequence length: T=10 → 5.

2. BiLSTM block — captures long-range bidirectional dependencies.
   Output from CNN is permuted back to (batch, 5, 128) and fed into a 2-layer
   bidirectional LSTM.  Only the final timestep's hidden state is used,
   giving shape (batch, 256) — bidirectional doubles the hidden dimension.

3. Classifier head — Linear(256→128) + ReLU + Dropout + Linear(128→n_classes).
   Returns raw logits; nn.CrossEntropyLoss applies log_softmax internally,
   which is numerically more stable than an explicit softmax + NLLLoss.

Usage
-----
    model = LSTMCNN()
    logits = model(x)           # x: (batch, 10, 41)
    # logits: (batch, 5) — raw scores, no softmax applied
"""

import torch
import torch.nn as nn


class LSTMCNN(nn.Module):
    """Hybrid CNN + Bidirectional LSTM classifier for network traffic sequences.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep (default 41 for NSL-KDD).
    seq_len : int
        Number of timesteps in each window (default 10).
    n_classes : int
        Number of output classes (default 5: Normal/DoS/Probe/R2L/U2R).
    cnn_channels : tuple of int
        Out-channel sizes for the three Conv1d layers (default (64, 128, 128)).
    lstm_hidden : int
        Hidden size for the LSTM (default 128).  Bidirectional doubles this
        in the output, giving lstm_hidden * 2 features after the last timestep.
    lstm_layers : int
        Number of stacked LSTM layers (default 2).  Must be >= 2 for the
        inter-layer dropout to take effect; PyTorch emits a warning if
        lstm_layers == 1 and dropout > 0.
    dropout : float
        Dropout probability used in the LSTM (between layers) and in the
        classifier head (default 0.3).
    """

    def __init__(
        self,
        n_features: int = 41,
        seq_len: int = 10,
        n_classes: int = 5,
        cnn_channels: tuple = (64, 128, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        c1, c2, c3 = cnn_channels

        # ── CNN block ──────────────────────────────────────────────────────────
        # Operates on (batch, n_features, seq_len); MaxPool1d(2) halves seq_len.
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len: 10 → 5
        )

        # ── BiLSTM block ───────────────────────────────────────────────────────
        # Operates on (batch, seq_len//2, c3); output shape (batch, seq_len//2, lstm_hidden*2).
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,  # applied between layers; safe because lstm_layers=2
        )

        # ── Classifier head ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, F) — batch of sliding windows.

        Returns
        -------
        torch.Tensor
            Shape (batch, n_classes) — raw logits (no softmax).
        """
        # x: (batch, T=10, F=41)
        x = x.permute(0, 2, 1)          # → (batch, F=41, T=10)  Conv1d: (N, C_in, L)
        x = self.cnn(x)                  # → (batch, 128, 5)
        x = x.permute(0, 2, 1)          # → (batch, 5, 128)       LSTM:   (N, L, C)
        out, _ = self.lstm(x)            # → (batch, 5, 256)       bidirectional doubles hidden
        x = out[:, -1, :]               # → (batch, 256)           last timestep only
        return self.classifier(x)        # → (batch, n_classes)    raw logits
