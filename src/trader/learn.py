import gymnasium as gym
import pandas as pd
import numpy as np
from pathlib import Path

from stable_baselines3 import DQN

from trader.environment import CryptoTraderEnv

def generate_trending_sinusoid(
    num_days: int = 365,
    initial_price: float = 100.0,
    trend_per_day: float = 0.1,
    monthly_amplitude: float = 10.0,
    noise_std: float = 0.5
) -> pd.DataFrame:
    """
    Generates a sinusoidal wave with a 1-minute sample rate, a monthly frequency,
    and a linear upward trend.

    Args:
        num_days (int): The total number of days for the data.
        initial_price (float): The starting price of the series.
        trend_per_day (float): The average daily increase in price for the trend.
        monthly_amplitude (float): The amplitude of the monthly sine wave.
        noise_std (float): The standard deviation of the random noise to add.

    Returns:
        pd.DataFrame: A DataFrame with 'timestamp' and 'price' columns.
    """
    # 1. Define the time range at a 1-minute sample rate
    total_minutes = num_days * 24 * 60
    t = np.arange(total_minutes)

    # 2. Create the upward trend line
    trend_per_minute = trend_per_day / (24 * 60)
    trend = initial_price + t * trend_per_minute

    # 3. Create the sinusoidal component with a ~1-month frequency
    # Period T = number of minutes in a month
    minutes_in_month = 30 * 24 * 60
    sinusoid = monthly_amplitude * np.sin(2 * np.pi * t / minutes_in_month)

    # 4. Create some random noise
    noise = np.random.normal(0, noise_std, total_minutes)

    # 5. Combine the components
    prices = trend + sinusoid + noise

    # 6. Create a pandas DataFrame with a datetime index
    timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=total_minutes, freq='T'))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close_eur': prices
    })
    
    return df


if __name__ == '__main__':
    data_path = Path('data/')
    crypto_file = data_path / 'FORMEUR_1m_1095days.tsv'

    # Load the datasets
    print(f"Loading crypto data from: {crypto_file}")
    #crypto_prices_df = pd.read_csv(crypto_file, sep='\t') 
    crypto_prices_df = generate_trending_sinusoid(num_days=355)

    env = CryptoTraderEnv(
        initial_balance=1000,
        initial_crypto_holdings=0,
        crypto_prices=crypto_prices_df["close_eur"].values
    )

    # For DQN, hyperparameters like buffer_size, train_freq, and target_update_interval
    # are passed during initialization, not during .learn()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        batch_size=30,
        gradient_steps=60,
        gamma=0.999,
        train_freq=4,  # Update the model 
        target_update_interval=60*24, # Update the target network
        buffer_size=30*24*60, # Size of the replay buffer
        n_steps = 60,
        learning_starts=30*24*60, # Number of steps to collect before training starts
    )
    model.learn(
        total_timesteps=crypto_prices_df.shape[0] * 2, 
        log_interval=1, # Log training progress every 1 episodes
        progress_bar=True
    )
    model.save("dqn_crypto_trader")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn_crypto_trader")

    obs, info = env.reset()
    # Run one full episode to evaluate the trained agent
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    # Render the environment to see what the agent is doing
    env.render()

    print("\n--- Evaluation Finished ---")
    env.close()