"""Forecasting, confidence scoring and the convert-now-or-later decision.

Model (deliberately simple and transparent, because FX is very close to a
random walk and fancy models mostly overfit):

  * Log returns, drift = a *shrunk* version of the recent average return
    (``trend_shrink`` = 0 gives a pure random walk).
  * Volatility = EWMA (RiskMetrics) volatility that mean-reverts toward the
    long-run volatility over the horizon.
  * Shocks are bootstrapped from the pair's own standardized historical
    returns, so fat tails are preserved.
  * Monte Carlo paths give the fan chart and every probability shown in the UI.

Confidence is *measured*, not asserted: the same model is replayed over past
dates (walk-forward backtest) to check whether its intervals were calibrated
and whether its direction calls beat a coin flip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PHI = 0.97  # per-day speed at which short-term vol decays toward long-run vol
Z80, Z95 = 1.2816, 1.96
QUANTILES = [0.025, 0.10, 0.25, 0.5, 0.75, 0.90, 0.975]


# --------------------------------------------------------------------------
# Model pieces
# --------------------------------------------------------------------------
def _estimate(r: np.ndarray, lam: float, window: int, trend_shrink: float):
    """Return (daily drift, ewma variance, long-run variance) from log returns."""
    r = np.asarray(r, dtype=float)
    # EWMA variance, seeded with the first 20 squared returns
    var = np.var(r[:20]) if len(r) >= 20 else np.var(r)
    for x in r:
        var = lam * var + (1 - lam) * x * x
    long_var = np.var(r)
    mu = trend_shrink * float(np.mean(r[-window:]))
    return mu, var, long_var


def _var_path(ewma_var: float, long_var: float, horizon: int) -> np.ndarray:
    k = np.arange(1, horizon + 1)
    return long_var + (ewma_var - long_var) * PHI**k


def _standardized_residuals(r: np.ndarray, lam: float) -> np.ndarray:
    var = np.var(r[:20]) if len(r) >= 20 else np.var(r)
    z = np.empty(len(r))
    for i, x in enumerate(r):
        z[i] = x / np.sqrt(var)
        var = lam * var + (1 - lam) * x * x
    z = z[20:] if len(z) > 60 else z
    z = (z - z.mean()) / z.std()
    return z


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------
@dataclass
class Forecast:
    last_date: pd.Timestamp
    last_rate: float
    dates: pd.DatetimeIndex
    paths: np.ndarray  # (n_paths, horizon) of rates
    bands: pd.DataFrame  # index=dates, columns=QUANTILES
    p_higher_by_day: np.ndarray  # P(rate_t > today) for each day
    daily_vol: float
    mu_daily: float


def forecast(series: pd.Series, horizon: int, n_paths: int = 5000, lam: float = 0.94,
             window: int = 60, trend_shrink: float = 0.25, seed: int = 7) -> Forecast:
    logp = np.log(series.values.astype(float))
    r = np.diff(logp)
    mu, ewma_var, long_var = _estimate(r, lam, window, trend_shrink)
    var_k = _var_path(ewma_var, long_var, horizon)
    z = _standardized_residuals(r, lam)

    rng = np.random.default_rng(seed)
    shocks = rng.choice(z, size=(n_paths, horizon), replace=True)
    steps = mu + np.sqrt(var_k)[None, :] * shocks
    paths = np.exp(logp[-1] + np.cumsum(steps, axis=1))

    last_date = series.index[-1]
    dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    bands = pd.DataFrame(np.quantile(paths, QUANTILES, axis=0).T, index=dates, columns=QUANTILES)
    last_rate = float(series.iloc[-1])
    return Forecast(
        last_date=last_date,
        last_rate=last_rate,
        dates=dates,
        paths=paths,
        bands=bands,
        p_higher_by_day=(paths > last_rate).mean(axis=0),
        daily_vol=float(np.sqrt(ewma_var)),
        mu_daily=mu,
    )


# --------------------------------------------------------------------------
# Walk-forward backtest (measures how trustworthy the forecasts are)
# --------------------------------------------------------------------------
@dataclass
class Backtest:
    n: int
    n_eff: float  # effective independent samples (windows overlap)
    hit_rate: float | None
    coverage80: float
    coverage95: float
    skill_vs_naive: float  # 1 - MAE(model)/MAE(no-change); >0 means beats "rate stays the same"
    detail: pd.DataFrame = field(repr=False)


def backtest(series: pd.Series, horizon: int, lam: float = 0.94, window: int = 60,
             trend_shrink: float = 0.25, step: int = 3, min_train: int = 120) -> Backtest | None:
    logp = np.log(series.values.astype(float))
    r = np.diff(logp)
    rows = []
    for t in range(min_train, len(logp) - horizon, step):
        mu, ev, lv = _estimate(r[:t], lam, window, trend_shrink)
        sd_h = float(np.sqrt(_var_path(ev, lv, horizon).sum()))
        mh = mu * horizon
        actual = logp[t + horizon] - logp[t]
        rows.append({
            "date": series.index[t],
            "pred": mh,
            "actual": actual,
            "in80": abs(actual - mh) <= Z80 * sd_h,
            "in95": abs(actual - mh) <= Z95 * sd_h,
            "hit": (np.sign(mh) == np.sign(actual)) if mh != 0 else np.nan,
        })
    if len(rows) < 15:
        return None
    df = pd.DataFrame(rows)
    mae_model = float(np.mean(np.abs(df.actual - df.pred)))
    mae_naive = float(np.mean(np.abs(df.actual)))
    hit = df["hit"].dropna()
    return Backtest(
        n=len(df),
        n_eff=max(1.0, len(df) * step / horizon),
        hit_rate=float(hit.mean()) if len(hit) else None,
        coverage80=float(df.in80.mean()),
        coverage95=float(df.in95.mean()),
        skill_vs_naive=1 - mae_model / mae_naive if mae_naive > 0 else 0.0,
        detail=df,
    )


# --------------------------------------------------------------------------
# Confidence score
# --------------------------------------------------------------------------
@dataclass
class Confidence:
    score: float  # 0-100
    label: str
    signal: float
    reliability: float
    calibration: float
    notes: list[str]


def confidence(p_up_h: float, bt: Backtest | None) -> Confidence:
    """Blend three 0-1 ingredients into one 0-100 score.

    signal       how lopsided today's forecast is (P(up) far from 50%)
    reliability  did past direction calls beat a coin flip (binomial z-score,
                 using the effective sample size)
    calibration  did past 80% intervals actually contain ~80% of outcomes
    """
    notes = []
    signal = min(1.0, abs(p_up_h - 0.5) / 0.2)

    reliability, calibration = 0.0, 0.5
    if bt is not None:
        if bt.hit_rate is not None:
            z = (bt.hit_rate - 0.5) / np.sqrt(0.25 / bt.n_eff)
            reliability = float(np.clip(z / 2.5, 0, 1))
            notes.append(f"Past direction calls were right {bt.hit_rate:.0%} of the time "
                         f"(about {bt.n_eff:.0f} independent tests; a coin flip is 50%).")
        else:
            notes.append("Trend influence is 0%, so there is no directional call to backtest.")
        calibration = float(np.clip(1 - abs(bt.coverage80 - 0.8) / 0.2, 0, 1))
        notes.append(f"The model's 80% range contained the real outcome {bt.coverage80:.0%} "
                     f"of the time in the backtest (ideal: 80%).")
        notes.append(f"Versus simply assuming the rate stays flat, the model's error was "
                     f"{'lower' if bt.skill_vs_naive > 0 else 'higher'} by {abs(bt.skill_vs_naive):.0%}.")
    else:
        notes.append("Not enough history to backtest; confidence is understated.")

    score = 100 * (0.45 * signal + 0.35 * reliability + 0.20 * calibration)
    label = "Low" if score < 35 else "Moderate" if score < 60 else "High"
    return Confidence(score, label, signal, reliability, calibration, notes)


# --------------------------------------------------------------------------
# Convert now vs later
# --------------------------------------------------------------------------
@dataclass
class Advice:
    verdict: str  # "CONVERT NOW" | "WAIT" | "NO CLEAR EDGE"
    headline: str
    reasons: list[str]
    p_up: float
    expected_change: float  # fraction
    median_change: float
    downside5: float  # 5th percentile change, fraction
    upside95: float


def advise(fc: Forecast, conf: Confidence, horizon_days: int, min_edge: float = 0.0025,
           p_threshold: float = 0.58) -> Advice:
    """User holds the *base* currency and wants the *quote* currency, so a higher
    base/quote rate is better for them."""
    final = fc.paths[:, -1]
    chg = final / fc.last_rate - 1
    p_up = float((chg > 0).mean())
    exp_chg = float(chg.mean())
    med_chg = float(np.median(chg))
    d5, u95 = float(np.quantile(chg, 0.05)), float(np.quantile(chg, 0.95))

    reasons = [
        f"Chance the rate is higher in {horizon_days} days than today: {p_up:.0%}.",
        f"Expected change: {exp_chg:+.2%} (median {med_chg:+.2%}).",
        f"Realistic range (90%): {d5:+.2%} to {u95:+.2%}.",
        f"Forecast confidence is {conf.label.lower()} ({conf.score:.0f}/100).",
    ]
    # deliberately conservative: wait only if the forecast is lopsided, the edge
    # exceeds the compensation you want for taking risk, and confidence isn't Low.
    if p_up >= p_threshold and exp_chg >= min_edge and conf.score >= 35:
        verdict = "WAIT"
        headline = "Waiting looks slightly favorable"
        reasons.append("Edge, probability and confidence all clear the thresholds. "
                       "If you can't tolerate the downside shown above, convert part now.")
    elif p_up <= 1 - p_threshold and exp_chg <= -min_edge:
        verdict = "CONVERT NOW"
        headline = "Converting now looks favorable"
        reasons.append("The model leans toward a lower rate later, so today's rate is the better bet.")
    else:
        verdict = "NO CLEAR EDGE"
        headline = "No reliable edge from waiting"
        reasons.append("Waiting adds risk without enough expected reward. If the timing is flexible, "
                       "converting now (or splitting into 2-3 chunks) is the lower-regret choice.")
    return Advice(verdict, headline, reasons, p_up, exp_chg, med_chg, d5, u95)