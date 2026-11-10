"""Deep-learning models (PyTorch, GPU if available): MLP, LSTM, GRU, Transformer over the last L days of features.

Leakage controls: the scaler is fitted on the training window only; early stopping uses the LAST 15% of the
training window (chronological), never the test year; each sequence ends at the decision day t (features known
at 09:15 on t) and never includes later rows.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, f, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(32, out))

    def forward(self, x):
        return self.net(x[:, -1, :])


class RNN(nn.Module):
    def __init__(self, f, out, cell="lstm"):
        super().__init__()
        self.rnn = (nn.LSTM if cell == "lstm" else nn.GRU)(f, 32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(32, out))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1, :])


class TinyTransformer(nn.Module):
    def __init__(self, f, out, L):
        super().__init__()
        self.inp = nn.Linear(f, 32)
        self.pos = nn.Parameter(torch.zeros(1, L, 32))
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.3, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(32, out)

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos[:, : x.shape[1]])
        return self.head(z[:, -1, :])


def make(name, f, L, out=1):
    return {"mlp": lambda: MLP(f, out), "lstm": lambda: RNN(f, out, "lstm"), "gru": lambda: RNN(f, out, "gru"),
            "transformer": lambda: TinyTransformer(f, out, L)}[name]()


def sequences(X: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    """Window ending at each row index (inclusive): rows idx-L+1 .. idx."""
    return np.stack([X[i - L + 1: i + 1] for i in idx])


def fit_predict(name: str, X: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray, task: str,
                L: int = 20, seed: int = 0, epochs: int = 80, patience: int = 10) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_idx = tr_idx[tr_idx >= L - 1]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-8                     # scaler on TRAIN rows only
    Xs = (X - mu) / sd
    cut = int(len(tr_idx) * 0.85)
    fit_i, val_i = tr_idx[:cut], tr_idx[cut:]
    ymu, ysd = (y[fit_i].mean(), y[fit_i].std() + 1e-8) if task == "reg" else (0.0, 1.0)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    Xf, Xv, Xt = t(sequences(Xs, fit_i, L)), t(sequences(Xs, val_i, L)), t(sequences(Xs, te_idx, L))
    yf, yv = t((y[fit_i] - ymu) / ysd), t((y[val_i] - ymu) / ysd)
    m = make(name, X.shape[1], L).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss() if task == "clf" else nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xf), device=DEV)
        for b in range(0, len(Xf), 64):
            j = perm[b:b + 64]
            opt.zero_grad()
            loss = lossf(m(Xf[j]).squeeze(-1), yf[j])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        with torch.no_grad():
            v = lossf(m(Xv).squeeze(-1), yv).item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        p = m(Xt).squeeze(-1)
        p = torch.sigmoid(p) if task == "clf" else p * ysd + ymu
    return p.cpu().numpy()
