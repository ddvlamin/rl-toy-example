"""Gymnasium environment for reinforcement learning cryptocurrency trading."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np

from trader.signals import compute_stationary_price_features

logger = logging.getLogger(__name__)

ENDOGENOUS_FEATURE_NAMES: tuple[str, ...] = (
    "position",
    "unrealized_pnl",
    "rel_to_last_sell",
    "holding_steps_norm",
)


class Actions(IntEnum):
    """Discrete trading actions available to the agent."""

    Hold = 0
    Buy = 1
    Sell = 2


class CryptoTraderEnv(gym.Env[np.ndarray, int]):
    """Cryptocurrency trading environment using stationary market and portfolio state.

    The observation vector concatenates exogenous stationary technical features
    precomputed from the price series (normalized rolling log-return window,
    zero-centered Wilder RSI, zero-centered Bollinger Band %B, and short/long
    SMA distances and spread) with endogenous portfolio features
    (``position``, ``unrealized_pnl``, ``rel_to_last_sell``,
    ``holding_steps_norm``).
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_balance: float = 1000.0,
        crypto_prices: Sequence[float] | np.ndarray | None = None,
        transaction_fee_percent: float = 0.01,
        log_ret_window: int = 10,
        *,
        zscore_window: int = 60,
        max_holding_steps: int | None = None,
        target_profit: float = 0.0,
        below_target_penalty: float = 0.0,
        holding_penalty: float = 0.0001,
        buy_above_last_sell_penalty: float = 0.01,
    ) -> None:
        """Initialize the trading environment and precompute price signal features.

        Args:
            initial_balance: Starting fiat balance (must be strictly positive).
            crypto_prices: 1D sequence of strictly positive asset prices.
            transaction_fee_percent: Proportional fee per trade in ``[0, 1)``.
            log_ret_window: Lookback horizon ``k`` for rolling normalized log returns.
            zscore_window: Trailing estimation window ``W`` (``W >= k``) for
                log-return Z-score standardization.
            max_holding_steps: Normalization horizon ``T_max`` for
                ``holding_steps_norm``. Defaults to ``episode_length``.
            target_profit: Net realized return threshold below which
                ``below_target_penalty`` is subtracted on trade exit.
            below_target_penalty: Non-negative penalty subtracted from exit reward
                when realized net return falls below ``target_profit``.
            holding_penalty: Non-negative step cost subtracted while holding an
                open position (``position == 1``).
            buy_above_last_sell_penalty: Non-negative shaping penalty subtracted
                when buying at ``P_t > P_last_sell``.
        """
        super().__init__()

        if crypto_prices is None:
            raise ValueError("crypto_prices must not be None.")
        self._validate_config(
            initial_balance=initial_balance,
            transaction_fee_percent=transaction_fee_percent,
            max_holding_steps=max_holding_steps,
            target_profit=target_profit,
            below_target_penalty=below_target_penalty,
            holding_penalty=holding_penalty,
            buy_above_last_sell_penalty=buy_above_last_sell_penalty,
        )

        raw_features, price_feature_names, start_step = (
            compute_stationary_price_features(
                crypto_prices,
                log_ret_window=log_ret_window,
                zscore_window=zscore_window,
            )
        )
        prices_arr = np.asarray(crypto_prices, dtype=np.float64)
        if prices_arr.shape[0] <= start_step + 1:
            raise ValueError(
                f"crypto_prices length ({prices_arr.shape[0]}) must exceed warmup "
                f"plus one step ({start_step + 1})."
            )

        self.initial_balance: float = float(initial_balance)
        self.crypto_prices: np.ndarray = prices_arr
        self.transaction_fee_percent: float = float(transaction_fee_percent)
        self.log_ret_window: int = log_ret_window
        self.zscore_window: int = zscore_window
        self.target_profit: float = float(target_profit)
        self.below_target_penalty: float = float(below_target_penalty)
        self.holding_penalty: float = float(holding_penalty)
        self.buy_above_last_sell_penalty: float = float(buy_above_last_sell_penalty)

        self.start_step: int = start_step
        self.episode_length: int = int(prices_arr.shape[0] - start_step - 1)
        self.max_holding_steps: int = (
            self.episode_length if max_holding_steps is None else max_holding_steps
        )
        self.price_features: np.ndarray = np.ascontiguousarray(
            raw_features, dtype=np.float32
        )
        self.feature_names: list[str] = [
            *price_feature_names,
            *ENDOGENOUS_FEATURE_NAMES,
        ]

        self.action_space = gym.spaces.Discrete(len(Actions))
        f32_bounds = np.finfo(np.float32)
        self.observation_space = gym.spaces.Box(
            low=f32_bounds.min,
            high=f32_bounds.max,
            shape=(len(self.feature_names),),
            dtype=np.float32,
        )

        self.balance: float = self.initial_balance
        self.crypto_holdings: float = 0.0
        self.current_step: int = self.start_step
        self.entry_price: float | None = None
        self.entry_step: int | None = None
        self.last_sell_price: float | None = None

    @staticmethod
    def _validate_config(
        initial_balance: float,
        transaction_fee_percent: float,
        max_holding_steps: int | None,
        target_profit: float,
        below_target_penalty: float,
        holding_penalty: float,
        buy_above_last_sell_penalty: float,
    ) -> None:
        """Validate environment financial, duration, and reward parameters."""
        if initial_balance <= 0.0 or not math.isfinite(initial_balance):
            raise ValueError(
                f"initial_balance must be finite and > 0, got {initial_balance}."
            )
        if not 0.0 <= transaction_fee_percent < 1.0:
            raise ValueError(
                "transaction_fee_percent must be in [0, 1), "
                f"got {transaction_fee_percent}."
            )
        if max_holding_steps is not None and max_holding_steps < 1:
            raise ValueError(
                f"max_holding_steps must be >= 1, got {max_holding_steps}."
            )
        if not math.isfinite(target_profit):
            raise ValueError(f"target_profit must be finite, got {target_profit}.")
        if below_target_penalty < 0.0 or not math.isfinite(below_target_penalty):
            raise ValueError(
                "below_target_penalty must be finite and >= 0, "
                f"got {below_target_penalty}."
            )
        if holding_penalty < 0.0 or not math.isfinite(holding_penalty):
            raise ValueError(
                f"holding_penalty must be finite and >= 0, got {holding_penalty}."
            )
        if buy_above_last_sell_penalty < 0.0 or not math.isfinite(
            buy_above_last_sell_penalty
        ):
            raise ValueError(
                "buy_above_last_sell_penalty must be finite and >= 0, "
                f"got {buy_above_last_sell_penalty}."
            )

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid discrete actions ``[Hold, Buy, Sell]``.

        Returns:
            Boolean ``ndarray`` of shape ``(3,)`` indicating legal actions for
            ``sb3_contrib.MaskablePPO``.
        """
        return np.array(
            [True, self.balance > 0.0, self.crypto_holdings > 0.0],
            dtype=bool,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the first post-warmup timestep.

        Args:
            seed: Optional random seed for the environment RNG.
            options: Optional reset configuration dictionary.

        Returns:
            Tuple of ``(initial_observation, info_dict)``.
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.crypto_holdings = 0.0
        self.current_step = self.start_step
        self.entry_price = None
        self.entry_step = None
        self.last_sell_price = None
        return self._get_observation(), {}

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one trading step and advance the market clock.

        Args:
            action: Discrete action index (``0=Hold``, ``1=Buy``, ``2=Sell``).

        Returns:
            Tuple ``(observation, reward, terminated, truncated, info)``.
        """
        if self.current_step >= self.crypto_prices.shape[0] - 1:
            raise RuntimeError(
                "Cannot call step() on a terminated episode; call reset() first."
            )

        reward = self._execute_trade(action)

        self.current_step += 1
        terminated = bool(self.current_step >= self.crypto_prices.shape[0] - 1)
        current_price = float(self.crypto_prices[self.current_step])

        if self.crypto_holdings > 0.0:
            reward -= self.holding_penalty
            if terminated and self.entry_price is not None:
                reward += self._compute_exit_reward(current_price, self.entry_price)

        current_portfolio_value = self._get_assets_value()
        obs = self._get_observation()
        info: dict[str, Any] = {
            "is_success": bool(current_portfolio_value > self.initial_balance),
            "portfolio_value": current_portfolio_value,
            "balance": self.balance,
            "crypto_holdings": self.crypto_holdings,
            "current_price": current_price,
            "entry_price": self.entry_price,
            "step": self.current_step,
        }
        return obs, float(reward), terminated, False, info

    def _compute_exit_reward(self, exit_price: float, entry_price: float) -> float:
        """Compute realized round-trip return after fees and target-profit penalty."""
        fee = self.transaction_fee_percent
        net_return = (
            exit_price * (1.0 - fee) - entry_price * (1.0 + fee)
        ) / entry_price
        if net_return < self.target_profit:
            return net_return - self.below_target_penalty
        return net_return

    def _execute_trade(self, action: int | np.integer) -> float:
        """Execute Buy/Sell/Hold order and return immediate trade reward/shaping."""
        try:
            parsed_action = Actions(int(action))
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}.") from exc

        current_price = float(self.crypto_prices[self.current_step])

        if parsed_action == Actions.Buy:
            if self.balance > 0.0:
                shaping = (
                    -self.buy_above_last_sell_penalty
                    if (
                        self.last_sell_price is not None
                        and current_price > self.last_sell_price
                    )
                    else 0.0
                )
                fee = self.transaction_fee_percent * self.balance
                buy_amount = self.balance - fee
                self.crypto_holdings += buy_amount / current_price
                self.balance = 0.0
                self.entry_price = current_price
                self.entry_step = self.current_step
                return shaping
            return 0.0

        if parsed_action == Actions.Sell:
            if self.crypto_holdings > 0.0 and self.entry_price is not None:
                exit_reward = self._compute_exit_reward(current_price, self.entry_price)
                sell_amount = self.crypto_holdings * current_price
                fee = self.transaction_fee_percent * sell_amount
                self.balance += sell_amount - fee
                self.crypto_holdings = 0.0
                self.entry_price = None
                self.entry_step = None
                self.last_sell_price = current_price
                return exit_reward
            return 0.0

        return 0.0

    def _get_observation(self) -> np.ndarray:
        """Assemble exogenous market features and endogenous portfolio features."""
        num_price_features = self.price_features.shape[1]
        obs = np.empty(len(self.feature_names), dtype=np.float32)
        obs[:num_price_features] = self.price_features[self.current_step]

        current_price = float(self.crypto_prices[self.current_step])
        in_position = self.crypto_holdings > 0.0
        position = 1.0 if in_position else 0.0
        unrealized_pnl = (
            (current_price - self.entry_price) / self.entry_price
            if in_position and self.entry_price is not None
            else 0.0
        )
        rel_to_last_sell = (
            (current_price - self.last_sell_price) / self.last_sell_price
            if self.last_sell_price is not None
            else 0.0
        )
        holding_steps_norm = (
            float(self.current_step - self.entry_step) / float(self.max_holding_steps)
            if in_position and self.entry_step is not None
            else 0.0
        )

        obs[num_price_features] = position
        obs[num_price_features + 1] = unrealized_pnl
        obs[num_price_features + 2] = rel_to_last_sell
        obs[num_price_features + 3] = holding_steps_norm
        return obs

    def _get_assets_value(self) -> float:
        """Compute total portfolio mark-to-market value in high-precision float."""
        current_price = float(self.crypto_prices[self.current_step])
        return self.balance + self.crypto_holdings * current_price

    def get_original_obs(self) -> np.ndarray:
        """Return the unnormalized observation vector for logging callbacks.

        Returns:
            Unnormalized ``float32`` observation vector at ``current_step``.
        """
        return self._get_observation()

    def render(self) -> None:
        """Log human-readable portfolio state at the current step."""
        current_price = float(self.crypto_prices[self.current_step])
        portfolio_value = self._get_assets_value()
        logger.info(
            "Step: %d | Balance: %.2f | Holdings: %.6f | Price: %.2f | Value: %.2f",
            self.current_step,
            self.balance,
            self.crypto_holdings,
            current_price,
            portfolio_value,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    dummy_prices = np.linspace(100.0, 110.0, 120) + rng.normal(0.0, 1.0, 120)
    demo_env = CryptoTraderEnv(crypto_prices=dummy_prices, log_ret_window=10)
    _obs, _info = demo_env.reset()
    episode_done = False
    total_reward = 0.0

    while not episode_done:
        sampled_action = demo_env.action_space.sample()
        _obs, step_reward, episode_done, _truncated, _info = demo_env.step(
            sampled_action
        )
        total_reward += step_reward
        demo_env.render()

    logger.info("Total reward: %.6f", total_reward)
