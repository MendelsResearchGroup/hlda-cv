from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ArrayLike = Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class HLDAResult:
    weights: pd.Series
    eigenvalue: float
    kept_descriptors: list[str]
    kept_indices: list[int]
    full_weights: dict[str, float] | None = None


def _as_2d_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like object")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    return array


def _validate_descriptor_names(desc_cols: Sequence[str], n_features: int) -> list[str]:
    names = list(desc_cols)
    if len(names) != n_features:
        raise ValueError(
            f"descriptor name count ({len(names)}) does not match feature count ({n_features})"
        )
    return names


def _covariance(X: np.ndarray) -> np.ndarray:
    cov = np.cov(X, rowvar=False, ddof=1)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov


def _regularize_covariance(cov: np.ndarray, ridge: float) -> np.ndarray:
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if ridge == 0:
        return cov
    return cov + ridge * np.eye(cov.shape[0], dtype=float)


def _spearman_abs_corr(X: np.ndarray) -> np.ndarray:
    if X.size == 0 or X.shape[0] < 2:
        return np.zeros((X.shape[1], X.shape[1]), dtype=float)
    corr, _ = spearmanr(X, axis=0)
    if np.isscalar(corr):
        return np.array([[1.0]], dtype=float)
    corr = np.abs(corr)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def prune(
    X_A: ArrayLike,
    X_B: ArrayLike,
    desc_cols: Sequence[str],
    threshold: float,
) -> tuple[list[str], list[int]]:
    """
    Drop highly correlated descriptors using Spearman rank correlation computed
    separately on each state and merged by union of drops.
    """
    XA = _as_2d_array(X_A, "X_A")
    XB = _as_2d_array(X_B, "X_B")
    names = _validate_descriptor_names(desc_cols, XA.shape[1])
    if XB.shape[1] != XA.shape[1]:
        raise ValueError("X_A and X_B must have the same number of features")

    corrA = _spearman_abs_corr(XA)
    corrB = _spearman_abs_corr(XB)
    corr = np.maximum(corrA, corrB)
    lower = np.tril(corr, k=-1)

    to_drop = [j for j in range(lower.shape[1]) if np.any(lower[:, j] > threshold)]
    drop_set = set(to_drop)
    keep_idx = [i for i in range(len(names)) if i not in drop_set]
    kept_cols = [names[i] for i in keep_idx]
    return kept_cols, keep_idx


def hlda_from_moments(
    muA: Sequence[float] | np.ndarray,
    SA: ArrayLike,
    muB: Sequence[float] | np.ndarray,
    SB: ArrayLike,
    desc_cols: Sequence[str],
    ridge: float = 1e-8,
) -> tuple[pd.Series, float]:
    muA = np.asarray(muA, dtype=float)
    muB = np.asarray(muB, dtype=float)
    SA = np.asarray(SA, dtype=float)
    SB = np.asarray(SB, dtype=float)
    names = _validate_descriptor_names(desc_cols, muA.shape[0])

    if muA.shape != muB.shape:
        raise ValueError("muA and muB must have the same shape")
    if SA.shape != SB.shape or SA.shape != (muA.shape[0], muA.shape[0]):
        raise ValueError("SA and SB must be square covariance matrices matching the descriptor count")

    SA_reg = _regularize_covariance(SA, ridge)
    SB_reg = _regularize_covariance(SB, ridge)

    d = muA - muB
    Sb = 0.5 * np.outer(d, d)
    Sw_inv = np.linalg.inv(SA_reg) + np.linalg.inv(SB_reg)

    eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
    idx = int(np.argmax(np.real(eigvals)))
    lam = float(np.real(eigvals[idx]))
    w = np.real(eigvecs[:, idx])

    Sw = np.linalg.inv(Sw_inv)
    w = w / np.sqrt(float(w.T @ Sw @ w))

    return pd.Series(w, index=names), lam


def complete_weights(
    desc_cols: Sequence[str],
    kept_cols: Sequence[str],
    weights_kept: pd.Series,
    covA: ArrayLike,
    covB: ArrayLike,
    keep_idx: Sequence[int],
) -> dict[str, float]:
    """
    Extend weights to dropped descriptors by mapping each dropped descriptor
    to the kept descriptor with the strongest correlation across both states.
    """
    full_weights = {name: float(w) for name, w in weights_kept.items()}
    keep_idx = list(keep_idx)
    names = list(desc_cols)
    kept_names = list(kept_cols)
    covA = np.asarray(covA, dtype=float)
    covB = np.asarray(covB, dtype=float)

    stdA_full = np.sqrt(np.diag(covA))
    stdB_full = np.sqrt(np.diag(covB))
    stdA_kept = stdA_full[keep_idx]
    stdB_kept = stdB_full[keep_idx]
    for j, desc in enumerate(names):
        if j in keep_idx:
            continue
        denomA = stdA_full[j] * stdA_kept
        denomB = stdB_full[j] * stdB_kept

        corrA = np.where(denomA > 0, covA[j, keep_idx] / denomA, 0.0)
        corrB = np.where(denomB > 0, covB[j, keep_idx] / denomB, 0.0)
        abs_comb = np.maximum(np.abs(corrA), np.abs(corrB))
        best = int(np.nanargmax(abs_comb))
        mapped_desc = kept_names[best]
        full_weights[desc] = float(weights_kept.get(mapped_desc, 0.0))

    return full_weights


def fit_hlda(
    X_A: ArrayLike,
    X_B: ArrayLike,
    desc_cols: Sequence[str],
    prune_threshold: float | None = None,
    include_pruned_weights: bool = False,
    ridge: float = 1e-8,
) -> HLDAResult:
    """
    Fit HLDA from descriptor matrices for two states.

    Parameters
    ----------
    X_A, X_B
        Samples x descriptors matrices for the two states.
    desc_cols
        Descriptor names in column order.
    prune_threshold
        Optional Spearman correlation threshold for descriptor pruning.
    include_pruned_weights
        When True and pruning is enabled, map dropped descriptors back onto the
        retained descriptor weights.
    ridge
        Diagonal covariance regularization added before matrix inversion.
    """
    XA = _as_2d_array(X_A, "X_A")
    XB = _as_2d_array(X_B, "X_B")
    names = _validate_descriptor_names(desc_cols, XA.shape[1])
    if XB.shape[1] != XA.shape[1]:
        raise ValueError("X_A and X_B must have the same number of features")

    kept_cols = names
    keep_idx = list(range(len(names)))
    if prune_threshold is not None:
        kept_cols, keep_idx = prune(XA, XB, names, threshold=prune_threshold)

    muA = XA[:, keep_idx].mean(axis=0)
    muB = XB[:, keep_idx].mean(axis=0)
    covA = _covariance(XA[:, keep_idx])
    covB = _covariance(XB[:, keep_idx])
    weights, eigenvalue = hlda_from_moments(muA, covA, muB, covB, kept_cols, ridge=ridge)

    full_weights = None
    if include_pruned_weights and prune_threshold is not None:
        full_weights = complete_weights(
            desc_cols=names,
            kept_cols=kept_cols,
            weights_kept=weights,
            covA=_covariance(XA),
            covB=_covariance(XB),
            keep_idx=keep_idx,
        )

    return HLDAResult(
        weights=weights,
        eigenvalue=eigenvalue,
        kept_descriptors=kept_cols,
        kept_indices=keep_idx,
        full_weights=full_weights,
    )
