import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss


def stabilized_ipw_binary(treatment, propensity, clip=1e-3):
    treatment = np.asarray(treatment).astype(int)
    propensity = np.clip(np.asarray(propensity), clip, 1.0 - clip)
    p_t = treatment.mean()
    numer = np.where(treatment == 1, p_t, 1.0 - p_t)
    denom = np.where(treatment == 1, propensity, 1.0 - propensity)
    return numer / denom


def weighted_mean(y, w):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    denom = np.sum(w)
    if denom <= 0:
        return np.nan
    return np.sum(w * y) / denom


def fit_propensity_binary(X, t, random_state=42):
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, t)
    p = model.predict_proba(X)[:, 1]
    return model, p


def doubly_robust_ate(X, treatment, outcome, random_state=42):
    treatment = np.asarray(treatment).astype(int)
    outcome = np.asarray(outcome, dtype=float)
    _, p = fit_propensity_binary(X, treatment, random_state=random_state)
    w = stabilized_ipw_binary(treatment, p)

    mu1 = LinearRegression().fit(X[treatment == 1], outcome[treatment == 1]) if np.any(treatment == 1) else None
    mu0 = LinearRegression().fit(X[treatment == 0], outcome[treatment == 0]) if np.any(treatment == 0) else None

    m1 = mu1.predict(X) if mu1 is not None else np.repeat(np.nan, len(outcome))
    m0 = mu0.predict(X) if mu0 is not None else np.repeat(np.nan, len(outcome))

    dr1 = m1 + treatment * (outcome - m1) / np.clip(p, 1e-3, 1.0)
    dr0 = m0 + (1 - treatment) * (outcome - m0) / np.clip(1 - p, 1e-3, 1.0)

    return {
        "ate_dr": float(np.nanmean(dr1 - dr0)),
        "treated_n": int(treatment.sum()),
        "control_n": int((1 - treatment).sum()),
        "ps_logloss": float(log_loss(treatment, np.clip(p, 1e-6, 1 - 1e-6))),
        "ipw_treated_mean": float(weighted_mean(outcome[treatment == 1], w[treatment == 1])) if np.any(treatment == 1) else np.nan,
        "ipw_control_mean": float(weighted_mean(outcome[treatment == 0], w[treatment == 0])) if np.any(treatment == 0) else np.nan,
    }
