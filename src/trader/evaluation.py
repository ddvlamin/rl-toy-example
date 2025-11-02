import pandas as pd
import numpy as np
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trader.environment import CryptoTraderEnv
from trader.sinusoid import generate_trending_sinusoid
from trader.signals import subsample_data
from trader.environment import Actions

if __name__ == '__main__':
    data_path = Path('data/')
    crypto_file = data_path / 'FORMEUR_1m_1095days.tsv'

    # Load the datasets
    print(f"Loading crypto data from: {crypto_file}")
    crypto_prices_df_1min = generate_trending_sinusoid(num_days=355)

    # Subsample the data to an hourly frequency
    print(f"Original data points: {len(crypto_prices_df_1min)}")
    crypto_prices_df = subsample_data(crypto_prices_df_1min, freq='D')
    print(f"Subsampled data points (daily): {len(crypto_prices_df)}")

    feature_window_size = 30

    # Create the base environment
    base_env = CryptoTraderEnv(
        initial_balance=1000,
        crypto_prices=crypto_prices_df["close_eur"].values,
        window_size=feature_window_size
    )

    # 1. Wrap the environment in a DummyVecEnv
    eval_vec_env = DummyVecEnv([lambda: base_env])
    # 2. Load the saved statistics and wrap the environment with VecNormalize
    # Use training=False to prevent the running average from being updated
    eval_env = VecNormalize.load("vec_normalize_stats.pkl", eval_vec_env)
    eval_env.training = False

    model = DQN.load("dqn_crypto_trader_2", env=eval_env)
    
    obs, info = eval_env.reset()
    # Run one full episode to evaluate the trained agent
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Action taken: {"Buy" if action == Actions.Buy else "Sell" if action == Actions.Sell else "Hold"}")
        obs, reward, terminated, truncated, info = eval_env.step(action)
        # The observation from a VecEnv is a batch of 1. We need to get the first element.
        obs_dict ={
            "norm_balance": float(obs[0][0]),
            "norm_holdings": float(obs[0][1]),
            "norm_avg_cost": float(obs[0][2]),
            "norm_avg_profit": float(obs[0][3]),
            "norm_price_start": float(obs[0][4]),
            "norm_price_trend": float(obs[0][5])
        }
        print(f"Normalized Observation: {obs_dict}")
        done = terminated or truncated
    # Render the environment to see what the agent is doing
    eval_env.render()

    print("\n--- Evaluation Finished ---")
    eval_env.close()
