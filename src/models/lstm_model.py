"""Single-layer LSTM forecast model using PyTorch.

Uses a direct (MIMO) forecasting strategy: one forward pass produces a
prediction for every requested horizon independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, n_outputs: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))


class LSTMModel:
    """Single-layer LSTM with direct multi-horizon forecasting.

    Parameters
    ----------
    hidden_size:
        Number of LSTM hidden units.
    context_window:
        Number of past observations used as input (look-back window).
    epochs:
        Training epochs.
    lr:
        Adam learning rate.
    horizons:
        List of forecast horizons for multi-output training.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        context_window: int = 24,
        epochs: int = 50,
        lr: float = 1e-3,
        horizons: list[int] | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.context_window = context_window
        self.epochs = epochs
        self.lr = lr
        self.horizons = sorted(horizons or [1, 6, 12, 24])
        self._net: _LSTMNet | None = None

    def _build_dataset(
        self, values: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slide a window over values to build (X, Y) pairs."""
        max_h = max(self.horizons)
        X_list, Y_list = [], []
        for i in range(self.context_window, len(values) - max_h + 1):
            x = values[i - self.context_window : i]
            y = np.array([values[i + h - 1] for h in self.horizons])
            X_list.append(x)
            Y_list.append(y)
        if not X_list:
            raise ValueError("Training series too short for the given context_window.")
        X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
        return X, Y

    def fit(self, train_series: pd.Series, scaler=None) -> "LSTMModel":
        """Train the LSTM on train_series.

        Parameters
        ----------
        train_series:
            Scaled (0-1) training series.
        scaler:
            Unused; kept for API compatibility with preprocessing.scale_fold.
        """
        torch.manual_seed(42)
        values = train_series.dropna().values.astype(float)
        X, Y = self._build_dataset(values)

        self._net = _LSTMNet(self.hidden_size, n_outputs=len(self.horizons))
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

        self._net.eval()
        return self

    def predict(self, context: np.ndarray, horizon: int) -> float:
        """Predict h steps ahead from a context window.

        Parameters
        ----------
        context:
            Last `context_window` scaled observations.
        horizon:
            Target horizon. Must be in self.horizons.

        Returns
        -------
        float
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        if horizon not in self.horizons:
            raise ValueError(f"horizon {horizon} not in trained horizons {self.horizons}.")

        ctx = np.asarray(context, dtype=float)[-self.context_window:]
        x = torch.tensor(ctx, dtype=torch.float32).view(1, -1, 1)
        with torch.no_grad():
            out = self._net(x).squeeze().numpy()
        idx = self.horizons.index(horizon)
        if out.ndim == 0:
            return float(out)
        return float(out[idx])
