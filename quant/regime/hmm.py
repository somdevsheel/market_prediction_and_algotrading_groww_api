"""Gaussian Hidden Markov Model (diagonal covariance), Baum-Welch EM with scaled forward-backward.

Implemented in NumPy so the algorithm is auditable. States are relabelled by ascending mean of the first
feature (e.g. returns) so labels are stable across refits. ``filtered_probs`` uses only past observations
(forward pass) and is what trading code must use; ``smoothed`` (forward-backward) is for research plots only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianHMM:
    n_states: int = 3
    n_iter: int = 200
    tol: float = 1e-5
    seed: int = 0
    min_var: float = 1e-6
    startprob: np.ndarray = field(default=None, repr=False)
    transmat: np.ndarray = field(default=None, repr=False)
    means: np.ndarray = field(default=None, repr=False)
    vars: np.ndarray = field(default=None, repr=False)
    loglik_: float = float("nan")
    converged_: bool = False
    mu_: np.ndarray = field(default=None, repr=False)   # standardisation
    sd_: np.ndarray = field(default=None, repr=False)

    def _std(self, X):
        return (X - self.mu_) / self.sd_

    def _emis(self, X):
        # log N(x | mean_k, diag var_k) for every t, k
        diff = X[:, None, :] - self.means[None, :, :]
        return -0.5 * (np.log(2 * np.pi * self.vars)[None] + diff ** 2 / self.vars[None]).sum(-1)

    def _forward(self, logB):
        T, K = logB.shape
        B = np.exp(logB - logB.max(1, keepdims=True))
        shift = logB.max(1)
        alpha = np.zeros((T, K))
        c = np.zeros(T)
        a = self.startprob * B[0]
        c[0] = a.sum() or 1e-300
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.transmat) * B[t]
            c[t] = a.sum() or 1e-300
            alpha[t] = a / c[t]
        return alpha, c, B, float(np.log(c).sum() + shift.sum())

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Z = self._std(X)
        T, D = Z.shape
        K = self.n_states
        rng = np.random.default_rng(self.seed)
        qs = np.quantile(Z[:, 0], np.linspace(0.15, 0.85, K))
        self.means = np.tile(Z.mean(0), (K, 1)) + rng.normal(0, 0.1, (K, D))
        self.means[:, 0] = qs
        self.vars = np.tile(Z.var(0), (K, 1))
        self.startprob = np.full(K, 1 / K)
        self.transmat = np.full((K, K), 0.1 / (K - 1)) + np.eye(K) * (0.9 - 0.1 / (K - 1))
        prev = -np.inf
        for it in range(self.n_iter):
            logB = self._emis(Z)
            alpha, c, B, ll = self._forward(logB)
            beta = np.ones((T, K))
            for t in range(T - 2, -1, -1):
                beta[t] = (self.transmat @ (B[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((K, K))
            for t in range(T - 1):
                x = alpha[t][:, None] * self.transmat * (B[t + 1] * beta[t + 1])[None] / c[t + 1]
                xi += x / (x.sum() or 1e-300)
            self.startprob = gamma[0] + 1e-12
            self.startprob /= self.startprob.sum()
            self.transmat = xi + 1e-6
            self.transmat /= self.transmat.sum(1, keepdims=True)
            w = gamma.sum(0)[:, None] + 1e-12
            self.means = (gamma.T @ Z) / w
            self.vars = np.maximum((gamma.T @ Z ** 2) / w - self.means ** 2, self.min_var)
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                self.converged_ = True
                break
            prev = ll
        self.loglik_ = ll
        order = np.argsort(self.means[:, 0])
        self.means, self.vars = self.means[order], self.vars[order]
        self.startprob = self.startprob[order]
        self.transmat = self.transmat[np.ix_(order, order)]
        return self

    def filtered_probs(self, X: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..x_t) — causal, safe for trading decisions."""
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        alpha, *_ = self._forward(self._emis(self._std(X)))
        return alpha

    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """P(state_{T+1} | x_1..x_T)."""
        return self.filtered_probs(X)[-1] @ self.transmat

    def bic(self, X) -> float:
        X = np.asarray(X, float)
        T = len(X)
        D = 1 if X.ndim == 1 else X.shape[1]
        K = self.n_states
        p = K - 1 + K * (K - 1) + 2 * K * D
        return -2 * self.loglik_ + p * np.log(T)

    def describe(self) -> list[dict]:
        m = self.means * self.sd_ + self.mu_
        s = np.sqrt(self.vars) * self.sd_
        return [{"state": k, "mean": m[k].round(5).tolist(), "std": s[k].round(5).tolist(),
                 "persistence": round(float(self.transmat[k, k]), 3)} for k in range(self.n_states)]


def label_states(model: GaussianHMM, vol_feature: int = 1) -> dict[int, str]:
    """Human labels: lowest-return state 'bearish', highest 'bullish'; if a vol feature exists, the highest-vol
    state among the rest is 'high_vol'."""
    K = model.n_states
    labels = {0: "hmm_bearish", K - 1: "hmm_bullish"}
    for k in range(1, K - 1):
        labels[k] = "hmm_neutral"
    if model.means.shape[1] > vol_feature and K >= 3:
        mid = list(range(1, K - 1))
        hv = max(mid, key=lambda k: model.means[k, vol_feature])
        labels[hv] = "hmm_neutral_highvol" if K > 3 else labels[hv]
    return labels
