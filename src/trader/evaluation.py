"""Evaluation script for a trained MaskablePPO cryptocurrency trading agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trader.environment import Actions, CryptoTraderEnv
from trader.signals import subsample_data
from trader.sinusoid import generate_trending_sinusoid

logger = logging.getLogger(__name__)


def main() -> None:
    """Evaluate a saved MaskablePPO policy over one full episode."""
    logging.basicConfig(level=logging.INFO)
    data_path = Path("data/")
    crypto_file = data_path / "FORMEUR_1m_1095days.tsv"

    logger.info("Generating evaluation price data (reference path: %s)", crypto_file)
    crypto_prices_df_1min = generate_trending_sinusoid(num_days=355)
    crypto_prices_df = subsample_data(crypto_prices_df_1min, freq="D")

    base_env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=crypto_prices_df["close_eur"].to_numpy(),
        log_ret_window=10,
        zscore_window=60,
    )

    eval_vec_env = DummyVecEnv([lambda: base_env])
    eval_env = VecNormalize.load("vec_normalize_stats.pkl", eval_vec_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = MaskablePPO.load("maskable_ppo_crypto_trader", env=eval_env)

    obs = eval_env.reset()
    done = False
    terminal_info: dict[str, Any] = {}
    while not done:
        masks = get_action_masks(eval_env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        action_enum = Actions(int(action[0]))
        step_obs, _, dones, infos = eval_env.step(action)
        if not isinstance(step_obs, np.ndarray):
            raise TypeError("Expected Box ndarray observation from VecNormalize.")
        obs = step_obs
        done = bool(dones[0])
        terminal_info = infos[0]
        obs_vec = terminal_info.get("terminal_observation", obs[0]) if done else obs[0]
        obs_dict = dict(
            zip(base_env.feature_names, (float(val) for val in obs_vec), strict=True)
        )
        logger.info("Action: %s | Normalized Obs: %s", action_enum.name, obs_dict)

    logger.info(
        "Evaluation finished | Step: %s | Balance: %.2f | Holdings: %.6f | "
        "Price: %.2f | Final Portfolio Value: %.2f",
        terminal_info.get("step"),
        float(terminal_info.get("balance", 0.0)),
        float(terminal_info.get("crypto_holdings", 0.0)),
        float(terminal_info.get("current_price", 0.0)),
        float(terminal_info.get("portfolio_value", 0.0)),
    )
    eval_env.close()


if __name__ == "__main__":
    main()
