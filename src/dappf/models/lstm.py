"""PyTorch Multi-Horizon Sequence-to-Sequence LSTM Forecaster."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from loguru import logger


class SequenceDataset(Dataset):
    """Dataset creating (168h lookback -> 24h forecast) sequence pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 168, horizon: int = 24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.valid_indices = range(len(X) - seq_len - horizon + 1)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = start + self.seq_len
        x_seq = self.X[start:end]
        y_horizon = self.y[end : end + self.horizon]
        return x_seq, y_horizon


class MultiHorizonLSTMNet(nn.Module):
    """2-Layer LSTM with 128 hidden units, dropout 0.2, and 24-horizon linear projection head."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, horizon: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        out, (hn, cn) = self.lstm(x)
        # Take the hidden state of the final time step
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        preds = self.head(last_hidden)  # [batch_size, 24]
        return preds


class LSTMForecaster:
    """Manages scaling, PyTorch training, early stopping, and multi-horizon inference."""

    def __init__(
        self,
        seq_len: int = 168,
        horizon: int = 24,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 20,
        patience: int = 7,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        # Detect acceleration device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.scaler = StandardScaler()
        self.net: Optional[MultiHorizonLSTMNet] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        val_fraction: float = 0.15,
    ) -> "LSTMForecaster":
        """Fit scaler on train data only, split validation tail, and train with early stopping."""
        X_train = X_train.bfill().ffill().fillna(0.0)
        y_train = y_train.bfill().ffill().fillna(0.0)
        self.feature_names = list(X_train.columns)
        n_train = len(X_train)
        split_idx = int(n_train * (1.0 - val_fraction))

        # Scale features strictly with training set statistics
        X_tr_raw = X_train.iloc[:split_idx].values
        X_val_raw = X_train.iloc[split_idx:].values
        self.scaler.fit(X_tr_raw)

        X_tr = self.scaler.transform(X_tr_raw)
        X_val = self.scaler.transform(X_val_raw)

        y_tr = y_train.iloc[:split_idx].values
        y_val = y_train.iloc[split_idx:].values

        train_ds = SequenceDataset(X_tr, y_tr, self.seq_len, self.horizon)
        val_ds = SequenceDataset(X_val, y_val, self.seq_len, self.horizon)

        if len(val_ds) == 0:
            # Fall back to training without validation split if small fold
            train_ds = SequenceDataset(self.scaler.transform(X_train.values), y_train.values, self.seq_len, self.horizon)
            val_ds = None

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False) if val_ds else None

        input_size = X_train.shape[1]
        self.net = MultiHorizonLSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)

        criterion = nn.L1Loss()  # Optimizes MAE directly
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        logger.info(f"Training LSTM on {self.device} (max epochs: {self.max_epochs}, patience: {self.patience})...")

        for epoch in range(self.max_epochs):
            self.net.train()
            train_losses = []
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                preds = self.net(bx)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            if val_loader:
                self.net.eval()
                val_losses = []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        preds = self.net(bx)
                        loss = criterion(preds, by)
                        val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.3f}")
                    break
            else:
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

        if best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict(self, X_full: pd.DataFrame, test_start_idx: int) -> np.ndarray:
        """
        Generate 24h rolling block forecasts across the test sequence.

        X_full contains warm-up history + test data.
        test_start_idx marks the beginning of the out-of-sample test window.
        """
        self.net.eval()
        X_scaled = self.scaler.transform(X_full.values)
        n_test = len(X_full) - test_start_idx
        preds_full = np.zeros(n_test)

        # Step through test set in daily 24h blocks
        with torch.no_grad():
            curr = test_start_idx
            while curr < len(X_full):
                # We need past 168 hours up to curr
                lookback_start = max(0, curr - self.seq_len)
                seq = X_scaled[lookback_start:curr]
                if len(seq) < self.seq_len:
                    # Pad if near edge
                    pad = np.repeat(seq[:1], self.seq_len - len(seq), axis=0)
                    seq = np.vstack([pad, seq])
                
                bx = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_24 = self.net(bx).cpu().numpy().flatten()

                n_assign = min(self.horizon, len(X_full) - curr)
                offset = curr - test_start_idx
                preds_full[offset : offset + n_assign] = pred_24[:n_assign]
                curr += self.horizon

        return preds_full
