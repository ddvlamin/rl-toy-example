"""Training entrypoint for the DQN cryptocurrency trading agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trader.callbacks import EpisodeLogger, EpisodeSummaryCallback
from trader.custom_dqn import LoggingDQN
from trader.environment import CryptoTraderEnv
from trader.signals import subsample_data
from trader.sinusoid import generate_trending_sinusoid

logger = logging.getLogger(__name__)


def main() -> None:
    """Train LoggingDQN on stationary price features and save artifacts."""
    logging.basicConfig(level=logging.INFO)
    data_path = Path("data/")
    crypto_file = data_path / "FORMEUR_1m_1095days.tsv"
    log_dir = Path("logs/")
    os.makedirs(log_dir, exist_ok=True)

    sb3_logger = configure(str(log_dir), ["stdout", "csv"])

    logger.info("Generating synthetic price data (reference path: %s)", crypto_file)
    crypto_prices_df_1min = generate_trending_sinusoid(num_days=355)

    logger.info("Original data points: %d", len(crypto_prices_df_1min))
    crypto_prices_df = subsample_data(crypto_prices_df_1min, freq="D")
    logger.info("Subsampled data points (daily): %d", len(crypto_prices_df))

    base_env = CryptoTraderEnv(
        initial_balance=1000.0,
        crypto_prices=crypto_prices_df["close_eur"].to_numpy(),
        log_ret_window=10,
        zscore_window=60,
        invalid_action_penalty=0.01,
    )
    episode_length = base_env.episode_length

    monitored_env: Monitor[np.ndarray, int] = Monitor(
        base_env, str(log_dir), info_keywords=("is_success",)
    )
    vec_env = DummyVecEnv([lambda: monitored_env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    policy_kwargs = {"net_arch": [64, 64]}
    model = LoggingDQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        gradient_steps=1,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        buffer_size=100000,
        learning_starts=1000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    model.set_logger(sb3_logger)

    num_episodes_to_train = 5000
    total_timesteps_estimate = num_episodes_to_train * episode_length

    episode_log_callback = EpisodeSummaryCallback(verbose=1)
    episode_logger = EpisodeLogger(logfile=log_dir / "episode_steps_log.tsv")
    callback_list = CallbackList([episode_logger, episode_log_callback])

    logger.info(
        "Training for %d episodes (%d timesteps).",
        num_episodes_to_train,
        total_timesteps_estimate,
    )
    model.learn(
        total_timesteps=total_timesteps_estimate,
        log_interval=10,
        callback=callback_list,
    )

    model.save("dqn_crypto_trader")
    env.save("vec_normalize_stats.pkl")
    logger.info("Training complete. Model saved as 'dqn_crypto_trader'.")


if __name__ == "__main__":
    main()
