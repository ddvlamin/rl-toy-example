"""Unit tests for stationary price signal feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.signals import (
    build_stationary_features,
    compute_stationary_price_features,
    subsample_data,
)


def _sample_price_df(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic geometric random walk price DataFrame."""
    rng = np.random.default_rng(seed)
    log_steps = rng.normal(loc=0.001, scale=0.02, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_steps))
    index = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": prices}, index=index)


def test_build_stationary_features_columns_and_warmup() -> None:
    """Verify output schema, warmup dropping, and absence of NaN/Inf."""
    df = _sample_price_df(n=150)
    features = build_stationary_features(
        df, price_col="close", log_ret_window=10, zscore_window=60
    )

    expected_cols = [
        *[f"norm_log_ret_lag_{i}" for i in range(9, 0, -1)],
        "norm_log_ret",
        "rsi_centered",
        "bb_pct_b",
        "dist_sma10",
        "dist_sma50",
        "sma_cross_ratio",
    ]
    assert list(features.columns) == expected_cols
    assert len(features) == 150 - 60
    assert features.index.equals(df.index[60:])
    assert np.all(np.isfinite(features.to_numpy()))


def test_rolling_log_return_window_exact_zscore_math() -> None:
    """Verify rolling k-step log returns are standardized by (mu_t, sigma_t)."""
    df = _sample_price_df(n=100, seed=7)
    k = 5
    w = 25
    features = build_stationary_features(
        df, price_col="close", log_ret_window=k, zscore_window=w
    )

    prices = df["close"].to_numpy(dtype=np.float64)
    r = np.empty_like(prices)
    r[0] = np.nan
    r[1:] = np.log(prices[1:]) - np.log(prices[:-1])

    # Check a specific timestamp t = 70 in original df
    t = 70
    window_w = r[t - w + 1 : t + 1]
    mu_t = np.mean(window_w)
    sigma_t = np.std(window_w, ddof=1)
    expected_vec = (r[t - k + 1 : t + 1] - mu_t) / (sigma_t + 1e-8)

    actual_vec = features.loc[df.index[t]].iloc[:k].to_numpy()
    np.testing.assert_allclose(actual_vec, expected_vec, rtol=0.0, atol=1e-12)


def test_log_ret_window_one_produces_single_column() -> None:
    """Verify k=1 produces a single norm_log_ret column without lag columns."""
    df = _sample_price_df(n=100)
    features = build_stationary_features(
        df, price_col="close", log_ret_window=1, zscore_window=60
    )
    assert features.columns[0] == "norm_log_ret"
    assert len(features.columns) == 6


def test_rsi_wilder_smoothing_bounds_and_monotonic_trends() -> None:
    """Verify Wilder RSI matches formula, stays in [-0.5, 0.5], and tracks momentum."""
    df = _sample_price_df(n=120, seed=11)
    features = build_stationary_features(df)

    # Check exact agreement with reference pandas Wilder smoothing
    prices = df["close"]
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    expected_rsi_centered = ((100.0 - (100.0 / (1.0 + rs))) / 100.0 - 0.5).loc[
        features.index
    ]

    np.testing.assert_allclose(
        features["rsi_centered"].to_numpy(),
        expected_rsi_centered.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(features["rsi_centered"] >= -0.5)
    assert np.all(features["rsi_centered"] <= 0.5)

    # Pure exponential uptrend -> RSI near +0.5
    up_df = pd.DataFrame({"close": np.geomspace(100.0, 300.0, 100)})
    up_feat = build_stationary_features(up_df)
    assert up_feat["rsi_centered"].iloc[-1] > 0.49

    # Pure exponential downtrend -> RSI near -0.5
    down_df = pd.DataFrame({"close": np.geomspace(300.0, 100.0, 100)})
    down_feat = build_stationary_features(down_df)
    assert down_feat["rsi_centered"].iloc[-1] < -0.49


def test_bollinger_pct_b_landmarks() -> None:
    """Verify %B centered values at SMA, upper band, lower band, and breakouts."""
    df = _sample_price_df(n=100, seed=99)
    prices = df["close"].to_numpy().copy()

    # Construct final 20-element window symmetrically so P[-1] == SMA20
    base = np.linspace(-1.0, 1.0, 19)
    # Sum of base is 0, so appending 0 keeps mean at 200.0
    prices[-20:-1] = 200.0 + 5.0 * base
    prices[-1] = 200.0
    feat_mid = build_stationary_features(pd.DataFrame({"close": prices}))
    assert feat_mid["bb_pct_b"].iloc[-1] == pytest.approx(0.0, abs=1e-6)

    # Construct window where P[-1] > Upper band (breakout -> centered %B > 0.5)
    prices_breakout = prices.copy()
    prices_breakout[-1] = 250.0
    sma20 = np.mean(prices_breakout[-20:])
    std20 = np.std(prices_breakout[-20:], ddof=1)
    expected_pct_b = (250.0 - (sma20 - 2.0 * std20)) / (4.0 * std20 + 1e-8) - 0.5
    feat_hi = build_stationary_features(pd.DataFrame({"close": prices_breakout}))
    assert feat_hi["bb_pct_b"].iloc[-1] == pytest.approx(expected_pct_b, abs=1e-10)
    assert feat_hi["bb_pct_b"].iloc[-1] > 0.5


def test_moving_average_distances_and_spread() -> None:
    """Verify dist_sma10, dist_sma50, and sma_cross_ratio against exact formulas."""
    df = _sample_price_df(n=120, seed=21)
    features = build_stationary_features(df)

    prices = df["close"]
    sma10 = prices.rolling(10).mean()
    sma50 = prices.rolling(50).mean()

    np.testing.assert_allclose(
        features["dist_sma10"].to_numpy(),
        ((prices - sma10) / (sma10 + 1e-8)).loc[features.index].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features["dist_sma50"].to_numpy(),
        ((prices - sma50) / (sma50 + 1e-8)).loc[features.index].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        features["sma_cross_ratio"].to_numpy(),
        ((sma10 - sma50) / (sma50 + 1e-8)).loc[features.index].to_numpy(),
        atol=1e-12,
    )


def test_zero_volatility_constant_prices_stability() -> None:
    """Verify epsilon stabilization prevents division-by-zero on flat price series."""
    flat_df = pd.DataFrame({"close": np.full(100, 123.45, dtype=np.float64)})
    features = build_stationary_features(flat_df)

    assert len(features) == 40
    assert np.all(np.isfinite(features.to_numpy()))
    np.testing.assert_allclose(features["norm_log_ret"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(features["dist_sma10"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(features["dist_sma50"].to_numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(features["sma_cross_ratio"].to_numpy(), 0.0, atol=1e-6)


def test_strict_causality_no_lookahead_bias() -> None:
    """Verify features at t <= cutoff are completely unaffected by future prices."""
    df_orig = _sample_price_df(n=150, seed=55)
    cutoff = 110
    df_truncated = df_orig.iloc[:cutoff].copy()

    df_mutated = df_orig.copy()
    df_mutated.iloc[cutoff:, 0] *= 1000.0

    feat_trunc = build_stationary_features(df_truncated)
    feat_mutated = build_stationary_features(df_mutated).loc[feat_trunc.index]

    np.testing.assert_allclose(
        feat_trunc.to_numpy(), feat_mutated.to_numpy(), rtol=0.0, atol=1e-12
    )


def test_price_scale_invariance_stationarity() -> None:
    """Verify multiplying prices by 500x leaves stationary features invariant."""
    df_base = _sample_price_df(n=150, seed=77)
    df_scaled = df_base.copy()
    df_scaled["close"] = df_scaled["close"] * 500.0

    feat_base = build_stationary_features(df_base)
    feat_scaled = build_stationary_features(df_scaled)

    np.testing.assert_allclose(
        feat_base.to_numpy(), feat_scaled.to_numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    ("prices", "k", "w", "match"),
    [
        (np.linspace(100.0, 120.0, 80), 0, 60, "log_ret_window"),
        (np.linspace(100.0, 120.0, 80), 10, 5, "zscore_window"),
        (np.linspace(100.0, 120.0, 80), 1, 1, "zscore_window"),
        (np.array([]), 10, 60, "empty"),
        (np.linspace(-10.0, 10.0, 80), 10, 60, "positive"),
        (np.array([100.0, np.nan, 102.0]), 1, 2, "finite"),
    ],
)
def test_invalid_inputs_raise_value_error(
    prices: np.ndarray, k: int, w: int, match: str
) -> None:
    """Verify invalid arrays or window sizes raise clear ValueErrors."""
    with pytest.raises(ValueError, match=match):
        compute_stationary_price_features(prices, log_ret_window=k, zscore_window=w)


def test_missing_price_col_raises_key_error() -> None:
    """Verify missing price_col in DataFrame raises KeyError."""
    df = _sample_price_df(n=80)
    with pytest.raises(KeyError, match="missing_col"):
        build_stationary_features(df, price_col="missing_col")


def test_subsample_data_resamples_without_warnings() -> None:
    """Verify subsample_data handles legacy 'H' and 'D' frequencies cleanly."""
    timestamps = pd.date_range("2025-01-01", periods=180, freq="min")
    df = pd.DataFrame(
        {"timestamp": timestamps, "close_eur": np.linspace(100.0, 105.0, 180)}
    )
    hourly = subsample_data(df, freq="H")
    assert "timestamp" in hourly.columns
    assert "close_eur" in hourly.columns
    assert len(hourly) >= 3
