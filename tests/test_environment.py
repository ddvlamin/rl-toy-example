"""Unit and integration tests for CryptoTraderEnv and RL training callbacks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trader.callbacks import (
    EpisodeLogger,
    EpisodeSummaryCallback,
    StopTrainingOnEpisodeCount,
)
from trader.custom_dqn import LoggingDQN
from trader.environment import Actions, CryptoTraderEnv
from trader.signals import build_stationary_features


def _make_prices(n: int = 120, seed: int = 123) -> np.ndarray:
    """Generate a strictly positive synthetic price trajectory."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.002, scale=0.015, size=n)
    return 100.0 * np.exp(np.cumsum(returns))


def test_gymnasium_env_checker_compliance() -> None:
    """Verify CryptoTraderEnv strictly complies with Gymnasium's env_checker."""
    prices = _make_prices(120)
    env = CryptoTraderEnv(crypto_prices=prices, log_ret_window=10, zscore_window=60)
    check_env(env, skip_render_check=True)


def test_no_legacy_or_unspecified_attributes() -> None:
    """Verify legacy/unspecified attributes and methods have been removed."""
    env = CryptoTraderEnv(crypto_prices=_make_prices(100))
    for removed_name in (
        "window_size",
        "avg_cost_of_holdings",
        "avg_profit",
        "last_action_price",
        "done",
        "include_portfolio_state",
        "action_masks",
    ):
        assert not hasattr(env, removed_name), f"Unexpected attribute: {removed_name}"


def test_observation_space_and_state_alignment_with_signals() -> None:
    """Verify env observations concatenate exogenous signals + 4 endogenous features."""
    prices = _make_prices(130, seed=9)
    df = pd.DataFrame({"close": prices})
    expected_price_df = build_stationary_features(
        df, price_col="close", log_ret_window=10, zscore_window=60
    )

    env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        log_ret_window=10,
        zscore_window=60,
        invalid_action_penalty=0.01,
    )
    assert env.observation_space.shape == (19,)
    assert env.observation_space.dtype == np.float32
    assert env.start_step == 60
    assert env.episode_length == 130 - 60 - 1
    assert env.max_holding_steps == env.episode_length
    assert env.feature_names[-4:] == [
        "position",
        "unrealized_pnl",
        "rel_to_last_sell",
        "holding_steps_norm",
    ]

    obs, info = env.reset()
    assert info == {}
    assert obs.shape == (19,)
    assert obs.dtype == np.float32

    np.testing.assert_allclose(
        obs[:15], expected_price_df.iloc[0].to_numpy(dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(obs[15:], np.zeros(4, dtype=np.float32))

    step_idx = 0
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, step_info = env.step(int(Actions.Hold))
        step_idx += 1
        assert isinstance(reward, float)
        assert not truncated
        assert "is_success" in step_info
        assert np.all(np.isfinite(obs))
        np.testing.assert_allclose(
            obs[:15],
            expected_price_df.iloc[step_idx].to_numpy(dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(obs[15:], np.zeros(4, dtype=np.float32))

    assert step_idx == env.episode_length


def test_endogenous_portfolio_features_and_action_safety() -> None:
    """Verify all 4 endogenous portfolio features and action safety."""
    prices = _make_prices(120, seed=42)
    env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        transaction_fee_percent=0.01,
        log_ret_window=10,
        zscore_window=60,
        invalid_action_penalty=0.05,
        max_holding_steps=100,
    )
    obs0, _ = env.reset()
    assert env.entry_price is None
    assert env.entry_step is None
    assert env.last_sell_price is None
    np.testing.assert_allclose(
        obs0[15:], np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    # 1. Invalid Sell while in cash: penalized, does NOT set last_sell_price
    obs_inv_sell, reward_inv_sell, _, _, _ = env.step(int(Actions.Sell))
    assert reward_inv_sell == pytest.approx(-0.05, abs=1e-9)
    np.testing.assert_allclose(
        obs_inv_sell[15:], np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    # 2. Valid Buy: enters position=1, tracks unrealized_pnl and holding_steps_norm
    entry_price = float(prices[env.current_step])
    obs_buy, _, _, _, _ = env.step(int(Actions.Buy))
    price_after_1 = float(prices[env.current_step])
    assert env.entry_price == pytest.approx(entry_price)
    assert obs_buy[15] == pytest.approx(1.0)
    assert obs_buy[16] == pytest.approx(
        (price_after_1 - entry_price) / entry_price, abs=1e-6
    )
    assert obs_buy[17] == pytest.approx(0.0)
    assert obs_buy[18] == pytest.approx(1.0 / 100.0, abs=1e-6)

    # 3. Hold for 10 more bars + 1 invalid Buy = 12 total bars in trade -> 0.12
    for _ in range(10):
        env.step(int(Actions.Hold))
    holdings_before = env.crypto_holdings
    prev_price = float(prices[env.current_step])
    obs_12, reward_inv_buy, _, _, _ = env.step(int(Actions.Buy))
    curr_price_12 = float(prices[env.current_step])
    assert reward_inv_buy == pytest.approx(
        np.log(curr_price_12 / prev_price) - 0.05, abs=1e-9
    )
    assert env.crypto_holdings == pytest.approx(holdings_before)
    assert env.entry_price == pytest.approx(entry_price)
    assert obs_12[15] == pytest.approx(1.0)
    assert obs_12[16] == pytest.approx(
        (curr_price_12 - entry_price) / entry_price, abs=1e-6
    )
    assert obs_12[17] == pytest.approx(0.0)
    assert obs_12[18] == pytest.approx(0.12, abs=1e-6)

    # 4. Valid Sell: resets position/pnl/duration and records last_sell_price
    sell_price = float(prices[env.current_step])
    obs_sell, _, _, _, _ = env.step(int(Actions.Sell))
    price_after_sell = float(prices[env.current_step])
    assert env.entry_price is None
    assert env.entry_step is None
    assert env.last_sell_price == pytest.approx(sell_price)
    assert obs_sell[15] == pytest.approx(0.0)
    assert obs_sell[16] == pytest.approx(0.0)
    assert obs_sell[17] == pytest.approx(
        (price_after_sell - sell_price) / sell_price, abs=1e-6
    )
    assert obs_sell[18] == pytest.approx(0.0)

    # 5. Re-Buy after previous Sell: both unrealized_pnl and rel_to_last_sell are active
    second_entry_price = float(prices[env.current_step])
    obs_rebuy, _, _, _, _ = env.step(int(Actions.Buy))
    price_after_rebuy = float(prices[env.current_step])
    assert obs_rebuy[15] == pytest.approx(1.0)
    assert obs_rebuy[16] == pytest.approx(
        (price_after_rebuy - second_entry_price) / second_entry_price, abs=1e-6
    )
    assert obs_rebuy[17] == pytest.approx(
        (price_after_rebuy - sell_price) / sell_price, abs=1e-6
    )
    assert obs_rebuy[18] == pytest.approx(0.01, abs=1e-6)

    # 6. Reset clears all endogenous trade memory
    obs_reset, _ = env.reset()
    assert env.entry_price is None
    assert env.entry_step is None
    assert env.last_sell_price is None
    np.testing.assert_allclose(obs_reset[15:], np.zeros(4, dtype=np.float32))

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


def test_invalid_env_configuration_raises_value_error() -> None:
    """Verify constructor validates crypto_prices length and financial parameters."""
    with pytest.raises(ValueError, match="crypto_prices"):
        CryptoTraderEnv(crypto_prices=None)

    with pytest.raises(ValueError, match="exceed warmup"):
        CryptoTraderEnv(crypto_prices=np.linspace(100.0, 110.0, 61), log_ret_window=10)

    with pytest.raises(ValueError, match="initial_balance"):
        CryptoTraderEnv(
            initial_balance=0.0, crypto_prices=_make_prices(100), log_ret_window=10
        )

    with pytest.raises(ValueError, match="max_holding_steps"):
        CryptoTraderEnv(
            crypto_prices=_make_prices(100), log_ret_window=10, max_holding_steps=0
        )


def test_terminal_info_metrics_and_post_termination_guard() -> None:
    """Verify step info preserves portfolio metrics and step() guards after done."""
    prices = np.linspace(100.0, 200.0, 70)
    env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        transaction_fee_percent=0.01,
        log_ret_window=10,
    )
    env.reset()

    _, _, terminated, _, info = env.step(int(Actions.Buy))
    while not terminated:
        _, _, terminated, _, info = env.step(int(Actions.Hold))

    assert info["portfolio_value"] > 1000.0
    assert info["crypto_holdings"] > 0.0
    assert info["balance"] == pytest.approx(0.0)
    assert info["current_price"] == pytest.approx(200.0)
    assert info["entry_price"] == pytest.approx(float(prices[60]))
    assert info["is_success"] is True

    with pytest.raises(RuntimeError, match="reset"):
        env.step(int(Actions.Hold))


def test_episode_logger_callback_and_sb3_dqn_integration(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify LoggingDQN + VecNormalize + EpisodeLogger + EpisodeSummaryCallback."""
    prices = np.linspace(100.0, 180.0, 100)
    base_env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        log_ret_window=10,
        invalid_action_penalty=0.01,
    )
    monitored: Monitor[np.ndarray, int] = Monitor(
        base_env, str(tmp_path), info_keywords=("is_success",)
    )
    vec_env = DummyVecEnv([lambda: monitored])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    log_file = tmp_path / "episode_steps_log.tsv"
    ep_logger = EpisodeLogger(logfile=log_file)
    assert not log_file.exists()

    ep_summary = EpisodeSummaryCallback(verbose=1)
    stop_cb = StopTrainingOnEpisodeCount(max_episodes=2, verbose=0)

    model = LoggingDQN(
        "MlpPolicy",
        norm_env,
        policy_kwargs={"net_arch": [32, 32]},
        learning_starts=20,
        batch_size=16,
        train_freq=4,
        gradient_steps=1,
        seed=1,
        verbose=0,
    )

    total_steps = base_env.episode_length * 2
    with caplog.at_level(logging.INFO, logger="trader.callbacks"):
        model.learn(
            total_timesteps=total_steps + 100,
            callback=CallbackList([ep_logger, ep_summary, stop_cb]),
        )

    logged_df = pd.read_csv(log_file, sep="\t")
    assert len(logged_df) == total_steps
    assert len(logged_df.columns) == 6 + 2 * len(base_env.feature_names)
    for col in (
        "orig_norm_log_ret",
        "obs_rsi_centered",
        "obs_bb_pct_b",
        "obs_sma_cross_ratio",
        "obs_position",
        "obs_unrealized_pnl",
        "obs_rel_to_last_sell",
        "obs_holding_steps_norm",
    ):
        assert col in logged_df.columns
    assert np.all(np.isfinite(logged_df.select_dtypes(include=[np.number]).to_numpy()))

    summary_msgs = [r.message for r in caplog.records if "Episode End" in r.message]
    assert len(summary_msgs) == 2
    assert any("PnL=0.00" not in msg for msg in summary_msgs)
