"""Unit and integration tests for CryptoTraderEnv, reward shaping, and MaskablePPO."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trader.callbacks import (
    EpisodeLogger,
    EpisodeSummaryCallback,
    StopTrainingOnEpisodeCount,
)
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
    """Verify legacy attributes have been removed."""
    env = CryptoTraderEnv(crypto_prices=_make_prices(100))
    for removed_name in (
        "window_size",
        "avg_cost_of_holdings",
        "avg_profit",
        "last_action_price",
        "last_portfolio_value",
        "invalid_action_penalty",
        "done",
        "include_portfolio_state",
    ):
        assert not hasattr(env, removed_name), f"Unexpected attribute: {removed_name}"


def test_observation_space_and_cash_hold_zero_reward() -> None:
    """Verify observations align with signals and cash Hold yields 0.0 reward."""
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
    )
    assert env.observation_space.shape == (19,)
    assert env.observation_space.dtype == np.float32
    assert env.start_step == 60
    assert env.episode_length == 130 - 60 - 1
    assert env.feature_names[-4:] == [
        "position",
        "unrealized_pnl",
        "rel_to_last_sell",
        "holding_steps_norm",
    ]

    obs, info = env.reset()
    assert info == {}
    np.testing.assert_allclose(
        obs[:15], expected_price_df.iloc[0].to_numpy(dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(obs[15:], np.zeros(4, dtype=np.float32))

    step_idx = 0
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, step_info = env.step(int(Actions.Hold))
        step_idx += 1
        assert reward == pytest.approx(0.0, abs=1e-12)
        assert not truncated
        assert "is_success" in step_info
        np.testing.assert_allclose(
            obs[:15],
            expected_price_df.iloc[step_idx].to_numpy(dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(obs[15:], np.zeros(4, dtype=np.float32))

    assert step_idx == env.episode_length


def test_realized_exit_reward_holding_penalty_and_target_profit() -> None:
    """Verify R_exit formula, holding_penalty, and below_target_penalty."""
    prices = np.full(80, 100.0, dtype=np.float64)
    # Add small variation in warmup so std > 0, then set exact trade prices
    prices[:60] = np.linspace(95.0, 100.0, 60)
    prices[60] = 100.0  # Buy at 100.0
    prices[61] = 102.0  # Hold in position
    prices[62] = 110.0  # Sell #1 at 110.0 -> net = (110*0.99 - 100*1.01)/100 = +0.079
    prices[63] = 100.0  # Buy #2 at 100.0 (<= 110.0, no buy-above-last-sell penalty)
    prices[64] = 105.0  # Sell #2 at 105.0 -> net = (105*0.99 - 100*1.01)/100 = +0.0295

    env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        transaction_fee_percent=0.01,
        log_ret_window=10,
        zscore_window=60,
        target_profit=0.05,
        below_target_penalty=0.02,
        holding_penalty=0.0001,
        buy_above_last_sell_penalty=0.01,
    )
    env.reset()
    np.testing.assert_array_equal(env.action_masks(), np.array([True, True, False]))

    # Step 60 -> 61: Buy at 100.0 -> holding penalty -0.0001
    _, rew_buy1, _, _, _ = env.step(int(Actions.Buy))
    assert rew_buy1 == pytest.approx(-0.0001, abs=1e-12)
    np.testing.assert_array_equal(env.action_masks(), np.array([True, False, True]))

    # Step 61 -> 62: Hold in position -> holding penalty -0.0001
    _, rew_hold1, _, _, _ = env.step(int(Actions.Hold))
    assert rew_hold1 == pytest.approx(-0.0001, abs=1e-12)

    # Step 62 -> 63: Sell at 110.0 -> net +0.079 >= 0.05 target -> no penalty
    _, rew_sell1, _, _, _ = env.step(int(Actions.Sell))
    assert rew_sell1 == pytest.approx(0.079, abs=1e-12)
    np.testing.assert_array_equal(env.action_masks(), np.array([True, True, False]))

    # Step 63 -> 64: Buy at 100.0 <= 110.0 last_sell -> only holding penalty -0.0001
    _, rew_buy2, _, _, _ = env.step(int(Actions.Buy))
    assert rew_buy2 == pytest.approx(-0.0001, abs=1e-12)

    # Step 64 -> 65: Sell at 105.0 -> net +0.0295 < 0.05 target -> 0.0095
    _, rew_sell2, _, _, _ = env.step(int(Actions.Sell))
    assert rew_sell2 == pytest.approx(0.0095, abs=1e-12)

    # Unmasked direct Sell while in cash is a safe no-op with 0.0 reward
    balance_after_sell2 = env.balance
    _, rew_noop_sell, _, _, _ = env.step(int(Actions.Sell))
    assert rew_noop_sell == pytest.approx(0.0, abs=1e-12)
    assert env.balance == pytest.approx(balance_after_sell2, abs=1e-12)


def test_condition_aware_shaping_buy_above_last_sell() -> None:
    """Verify soft penalty applies when buying at P_t > P_last_sell."""
    prices = np.linspace(95.0, 100.0, 70)
    prices[60] = 100.0  # Buy #1
    prices[61] = 105.0  # Sell #1 -> last_sell_price = 105.0
    prices[62] = 108.0  # Buy #2 at 108.0 > 105.0 -> incurs buy_above_last_sell_penalty

    env = CryptoTraderEnv(
        crypto_prices=prices,
        transaction_fee_percent=0.01,
        holding_penalty=0.0001,
        buy_above_last_sell_penalty=0.015,
    )
    env.reset()

    env.step(int(Actions.Buy))
    env.step(int(Actions.Sell))
    obs_rebuy, rew_rebuy, _, _, _ = env.step(int(Actions.Buy))

    assert rew_rebuy == pytest.approx(-0.015 - 0.0001, abs=1e-12)
    assert obs_rebuy[15] == pytest.approx(1.0)

    # Unmasked direct Buy while already holding is a safe no-op (only holding penalty)
    holdings_before = env.crypto_holdings
    entry_before = env.entry_price
    _, rew_noop_buy, _, _, _ = env.step(int(Actions.Buy))
    assert rew_noop_buy == pytest.approx(-0.0001, abs=1e-12)
    assert env.crypto_holdings == pytest.approx(holdings_before, abs=1e-12)
    assert env.entry_price == pytest.approx(entry_before, abs=1e-12)


def test_endogenous_portfolio_features_lifecycle() -> None:
    """Verify position, unrealized_pnl, rel_to_last_sell, and holding_steps_norm."""
    prices = _make_prices(120, seed=42)
    env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        transaction_fee_percent=0.01,
        log_ret_window=10,
        zscore_window=60,
        max_holding_steps=100,
    )
    obs0, _ = env.reset()
    np.testing.assert_allclose(obs0[15:], np.zeros(4, dtype=np.float32))

    entry_price = float(prices[env.current_step])
    obs_buy, _, _, _, _ = env.step(int(Actions.Buy))
    price_1 = float(prices[env.current_step])
    assert obs_buy[15] == pytest.approx(1.0)
    assert obs_buy[16] == pytest.approx((price_1 - entry_price) / entry_price, abs=1e-6)
    assert obs_buy[17] == pytest.approx(0.0)
    assert obs_buy[18] == pytest.approx(0.01, abs=1e-6)

    for _ in range(11):
        obs_12, _, _, _, _ = env.step(int(Actions.Hold))
    price_12 = float(prices[env.current_step])
    assert obs_12[15] == pytest.approx(1.0)
    assert obs_12[16] == pytest.approx((price_12 - entry_price) / entry_price, abs=1e-6)
    assert obs_12[18] == pytest.approx(0.12, abs=1e-6)

    sell_price = float(prices[env.current_step])
    obs_sell, _, _, _, _ = env.step(int(Actions.Sell))
    price_after_sell = float(prices[env.current_step])
    assert obs_sell[15] == pytest.approx(0.0)
    assert obs_sell[16] == pytest.approx(0.0)
    assert obs_sell[17] == pytest.approx(
        (price_after_sell - sell_price) / sell_price, abs=1e-6
    )
    assert obs_sell[18] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


def test_terminal_open_position_evaluates_r_exit_and_guards_post_done() -> None:
    """Verify open position at episode end receives terminal R_exit at P_T."""
    prices = np.linspace(90.0, 100.0, 65)
    prices[60] = 100.0  # Entry
    prices[64] = 90.0  # Final terminal price P_T (loss!)

    env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        transaction_fee_percent=0.01,
        target_profit=0.0,
        below_target_penalty=0.03,
        holding_penalty=0.0001,
    )
    env.reset()

    _, _, terminated, _, info = env.step(int(Actions.Buy))
    final_reward = 0.0
    while not terminated:
        _, final_reward, terminated, _, info = env.step(int(Actions.Hold))

    expected_net = (90.0 * 0.99 - 100.0 * 1.01) / 100.0  # -0.119 < 0.0
    expected_terminal_rew = -0.0001 + expected_net - 0.03
    assert final_reward == pytest.approx(expected_terminal_rew, abs=1e-12)
    assert info["entry_price"] == pytest.approx(100.0)

    with pytest.raises(RuntimeError, match="reset"):
        env.step(int(Actions.Hold))


def test_invalid_env_configuration_raises_value_error() -> None:
    """Verify constructor validates prices, fees, durations, and reward penalties."""
    with pytest.raises(ValueError, match="crypto_prices"):
        CryptoTraderEnv(crypto_prices=None)

    with pytest.raises(ValueError, match="exceed warmup"):
        CryptoTraderEnv(crypto_prices=np.linspace(100.0, 110.0, 61))

    with pytest.raises(ValueError, match="initial_balance"):
        CryptoTraderEnv(initial_balance=0.0, crypto_prices=_make_prices(100))

    with pytest.raises(ValueError, match="max_holding_steps"):
        CryptoTraderEnv(crypto_prices=_make_prices(100), max_holding_steps=0)

    with pytest.raises(ValueError, match="holding_penalty"):
        CryptoTraderEnv(crypto_prices=_make_prices(100), holding_penalty=-0.001)

    with pytest.raises(ValueError, match="below_target_penalty"):
        CryptoTraderEnv(crypto_prices=_make_prices(100), below_target_penalty=-0.01)

    with pytest.raises(ValueError, match="buy_above_last_sell_penalty"):
        CryptoTraderEnv(
            crypto_prices=_make_prices(100), buy_above_last_sell_penalty=-0.01
        )

    with pytest.raises(ValueError, match="target_profit"):
        CryptoTraderEnv(crypto_prices=_make_prices(100), target_profit=float("nan"))


def test_maskable_ppo_and_callbacks_integration(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify MaskablePPO + VecNormalize + EpisodeLogger + EpisodeSummaryCallback."""
    prices = np.linspace(100.0, 180.0, 100)
    base_env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=prices,
        log_ret_window=10,
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

    model = MaskablePPO(
        "MlpPolicy",
        norm_env,
        policy_kwargs={"net_arch": [32, 32]},
        n_steps=64,
        batch_size=32,
        n_epochs=2,
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
    assert np.all(np.isfinite(logged_df.select_dtypes(include=[np.number]).to_numpy()))

    summary_msgs = [r.message for r in caplog.records if "Episode End" in r.message]
    assert len(summary_msgs) == 2

    # Verify deterministic rollout with action_masks never selects illegal actions
    norm_env.training = False
    obs = norm_env.reset()
    done = False
    while not done:
        masks = get_action_masks(norm_env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        assert bool(masks[0, int(action[0])]) is True
        step_obs, _, dones, _ = norm_env.step(action)
        assert isinstance(step_obs, np.ndarray)
        obs = step_obs
        done = bool(dones[0])
