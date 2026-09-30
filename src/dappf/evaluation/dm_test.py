"""Diebold-Mariano Test with Harvey-Leybourne-Newbold (HLN) Small-Sample Correction."""

from typing import Dict, Literal, Tuple
import numpy as np
from scipy import stats


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 24,
    criterion: Literal["MAE", "MSE"] = "MAE",
    alternative: Literal["two_sided", "less", "greater"] = "two_sided",
) -> Dict[str, float]:
    """
    Compute the Diebold-Mariano test statistic with Harvey-Leybourne-Newbold (HLN) correction.

    Tests whether Model 1 is statistically significantly different/better than Model 2.
    Null hypothesis H0: E[d_t] = 0 (both models have identical predictive accuracy).
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth realized price series.
    y_pred1 : np.ndarray
        Forecasts from Model 1 (e.g. Challenger ML model).
    y_pred2 : np.ndarray
        Forecasts from Model 2 (e.g. Naive-24h benchmark).
    h : int
        Forecast horizon (e.g. 24 for day-ahead hourly forecast).
    criterion : "MAE" or "MSE"
        Loss differential criterion.
    alternative : "two_sided", "less", "greater"
        "less" tests if Model 1 has lower loss than Model 2.

    Returns
    -------
    dict:
        "dm_stat": raw DM statistic
        "hln_stat": HLN corrected DM statistic
        "p_value": p-value from Student's t distribution with (T - 1) degrees of freedom
        "is_significant": boolean whether p < 0.05
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred1 = np.asarray(y_pred1, dtype=np.float64)
    y_pred2 = np.asarray(y_pred2, dtype=np.float64)

    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    if criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "MSE":
        d = e1**2 - e2**2
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    T = len(d)
    if T < 2:
        return {"dm_stat": 0.0, "hln_stat": 0.0, "p_value": 1.0, "is_significant": False}

    d_mean = np.mean(d)

    # Auto-covariances gamma_k up to lag (h - 1)
    gamma_0 = np.var(d, ddof=0)
    gamma = []
    for k in range(1, h):
        cov_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)) if k < T else 0.0
        gamma.append(cov_k)

    # Long-run variance with Bartlett-type weighting
    lr_var = gamma_0 + 2.0 * sum((1.0 - (k + 1) / h) * gamma[k] for k in range(len(gamma)))
    lr_var = max(lr_var, 1e-8)

    dm_stat = d_mean / np.sqrt(lr_var / T)

    # HLN (1997) finite-sample correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    hln_stat = dm_stat * hln_factor

    # Degrees of freedom: T - 1
    df = max(1, T - 1)
    if alternative == "two_sided":
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(hln_stat), df=df))
    elif alternative == "less":
        p_val = stats.t.cdf(hln_stat, df=df)
    elif alternative == "greater":
        p_val = 1.0 - stats.t.cdf(hln_stat, df=df)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "dm_stat": float(dm_stat),
        "hln_stat": float(hln_stat),
        "p_value": float(np.clip(p_val, 0.0, 1.0)),
        "is_significant": bool(p_val < 0.05),
    }
