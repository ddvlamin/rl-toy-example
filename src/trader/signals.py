"""Signal processing and stationary feature engineering for price series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EPSILON: float = 1e-8
RSI_WINDOW: int = 14
BB_WINDOW: int = 20
BB_NUM_STD: float = 2.0
SMA_SHORT_WINDOW: int = 10
SMA_LONG_WINDOW: int = 50


def subsample_data(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Subsamples the price data to a lower frequency.

    Args:
        df: DataFrame with a 'timestamp' column or DatetimeIndex and prices.
        freq: The target pandas offset alias (e.g., 'h' for hourly, 'D' for daily).

    Returns:
        The subsampled DataFrame with a reset index.
    """
    target_freq = "h" if freq == "H" else freq
    working_df = df.set_index("timestamp") if "timestamp" in df.columns else df
    return working_df.resample(target_freq, label="right").last().dropna().reset_index()


def _validate_price_inputs(
    prices: Sequence[float] | np.ndarray,
    log_ret_window: int,
    zscore_window: int,
) -> np.ndarray:
    """Validate price series and rolling window parameters."""
    if log_ret_window < 1:
        raise ValueError(f"log_ret_window must be >= 1, got {log_ret_window}.")
    if zscore_window < 2:
        raise ValueError(f"zscore_window must be >= 2, got {zscore_window}.")
    if zscore_window < log_ret_window:
        raise ValueError(
            f"zscore_window ({zscore_window}) must be >= "
            f"log_ret_window ({log_ret_window})."
        )

    prices_arr = np.asarray(prices, dtype=np.float64)
    if prices_arr.ndim != 1 or prices_arr.size == 0:
        raise ValueError("prices must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(prices_arr)):
        raise ValueError("All prices must be finite (no NaN or Inf values).")
    if np.any(prices_arr <= 0.0):
        raise ValueError("All prices must be strictly positive.")
    return prices_arr


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute trailing rolling mean using vectorized NumPy sliding windows."""
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] >= window:
        result[window - 1 :] = sliding_window_view(values, window_shape=window).mean(
            axis=1
        )
    return result


def _rolling_std(values: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Compute trailing rolling std using vectorized NumPy sliding windows."""
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] >= window:
        result[window - 1 :] = sliding_window_view(values, window_shape=window).std(
            axis=1, ddof=ddof
        )
    return result


def _compute_normalized_log_returns(
    prices_arr: np.ndarray,
    log_ret_window: int,
    zscore_window: int,
) -> np.ndarray:
    """Compute k-step rolling log returns standardized by trailing W-step Z-score."""
    n = prices_arr.shape[0]
    log_ret = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        log_ret[1:] = np.diff(np.log(prices_arr))

    roll_mean = np.full(n, np.nan, dtype=np.float64)
    roll_std = np.full(n, np.nan, dtype=np.float64)
    if n > zscore_window:
        w_views = sliding_window_view(log_ret[1:], window_shape=zscore_window)
        roll_mean[zscore_window:] = w_views.mean(axis=1)
        roll_std[zscore_window:] = w_views.std(axis=1, ddof=1)

    r_window = np.full((n, log_ret_window), np.nan, dtype=np.float64)
    if n >= log_ret_window:
        r_window[log_ret_window - 1 :] = sliding_window_view(
            log_ret, window_shape=log_ret_window
        )

    return (r_window - roll_mean[:, np.newaxis]) / (roll_std[:, np.newaxis] + EPSILON)


def _compute_centered_rsi(prices_arr: np.ndarray) -> np.ndarray:
    """Compute zero-centered 14-period Wilder RSI in [-0.5, 0.5]."""
    prices_series = pd.Series(prices_arr, dtype=np.float64)
    delta = prices_series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / RSI_WINDOW
    avg_gain = gain.ewm(alpha=alpha, min_periods=RSI_WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=RSI_WINDOW, adjust=False).mean()

    rs = avg_gain.to_numpy(dtype=np.float64) / (
        avg_loss.to_numpy(dtype=np.float64) + EPSILON
    )
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return (rsi / 100.0) - 0.5


def compute_stationary_price_features(
    prices: Sequence[float] | np.ndarray,
    log_ret_window: int = 10,
    zscore_window: int = 60,
) -> tuple[np.ndarray, list[str], int]:
    """Compute stationary technical features from a 1D price signal.

    Constructs:
    1. Rolling window of ``log_ret_window`` 1-step log returns standardized via
       trailing ``zscore_window`` mean and sample standard deviation.
    2. Zero-centered 14-period Wilder RSI in ``[-0.5, 0.5]``.
    3. Zero-centered 20-period Bollinger Band %B (``%B - 0.5``).
    4. Fractional distances from 10/50-period SMAs and 10/50 SMA spread.

    Args:
        prices: Strictly positive, finite 1D sequence of asset prices.
        log_ret_window: Lookback horizon ``k`` for stacked log returns.
        zscore_window: Trailing estimation window ``W`` (``W >= k``) for Z-score.

    Returns:
        A tuple ``(feature_matrix, feature_names, start_step)`` where
        ``feature_matrix`` has shape ``(len(prices), log_ret_window + 5)`` with
        warmup rows ``[:start_step]`` containing ``NaN`` and rows ``[start_step:]``
        guaranteed finite.
    """
    prices_arr = _validate_price_inputs(prices, log_ret_window, zscore_window)

    norm_log_ret_window = _compute_normalized_log_returns(
        prices_arr, log_ret_window=log_ret_window, zscore_window=zscore_window
    )
    rsi_centered = _compute_centered_rsi(prices_arr)

    sma20 = _rolling_mean(prices_arr, BB_WINDOW)
    std20 = _rolling_std(prices_arr, BB_WINDOW, ddof=1)
    upper = sma20 + BB_NUM_STD * std20
    lower = sma20 - BB_NUM_STD * std20
    bb_pct_b = (prices_arr - lower) / (upper - lower + EPSILON) - 0.5

    sma10 = _rolling_mean(prices_arr, SMA_SHORT_WINDOW)
    sma50 = _rolling_mean(prices_arr, SMA_LONG_WINDOW)
    dist_sma10 = (prices_arr - sma10) / (sma10 + EPSILON)
    dist_sma50 = (prices_arr - sma50) / (sma50 + EPSILON)
    sma_cross_ratio = (sma10 - sma50) / (sma50 + EPSILON)

    feature_matrix = np.column_stack(
        [
            norm_log_ret_window,
            rsi_centered,
            bb_pct_b,
            dist_sma10,
            dist_sma50,
            sma_cross_ratio,
        ]
    )
    feature_names = [
        *[f"norm_log_ret_lag_{lag}" for lag in range(log_ret_window - 1, 0, -1)],
        "norm_log_ret",
        "rsi_centered",
        "bb_pct_b",
        "dist_sma10",
        "dist_sma50",
        "sma_cross_ratio",
    ]
    start_step = max(
        zscore_window,
        RSI_WINDOW,
        BB_WINDOW - 1,
        SMA_SHORT_WINDOW - 1,
        SMA_LONG_WINDOW - 1,
    )
    return feature_matrix, feature_names, start_step


def build_stationary_features(
    df: pd.DataFrame,
    price_col: str = "close",
    log_ret_window: int = 10,
    zscore_window: int = 60,
) -> pd.DataFrame:
    """Build stationary, zero-centered price features as a clean DataFrame.

    Args:
        df: Input DataFrame containing ``price_col``.
        price_col: Name of the column containing strictly positive prices.
        log_ret_window: Lookback horizon ``k`` for the rolling log-return vector.
        zscore_window: Trailing estimation window ``W`` (``W >= k``) for Z-score.

    Returns:
        DataFrame indexed like ``df`` with warmup ``NaN`` rows dropped.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in DataFrame.")

    feature_matrix, feature_names, _ = compute_stationary_price_features(
        df[price_col].to_numpy(dtype=np.float64),
        log_ret_window=log_ret_window,
        zscore_window=zscore_window,
    )
    features = pd.DataFrame(feature_matrix, index=df.index, columns=feature_names)
    return features.dropna()
