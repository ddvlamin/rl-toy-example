import gymnasium as gym
import pandas as pd
import numpy as np
from pathlib import Path

from stable_baselines3 import DQN

from trader.environment import CryptoTraderEnv, Actions
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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
    prices = trend + sinusoid #+ noise

    # 6. Create a pandas DataFrame with a datetime index
    timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=total_minutes, freq='T'))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close_eur': prices
    })
    
    return df

def subsample_data(df: pd.DataFrame, freq: str = 'H') -> pd.DataFrame:
    """
    Subsamples the price data to a lower frequency.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'close_eur' columns.
        freq (str): The target frequency (e.g., 'H' for hourly, 'D' for daily).

    Returns:
        pd.DataFrame: The subsampled DataFrame.
    """
    # Ensure 'timestamp' is the index for resampling
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Resample and take the last price in each new interval, then drop any NaNs
    df_resampled = df.resample(freq, label='right').last().dropna().reset_index()
    return df_resampled

class EpisodeSummaryCallback(BaseCallback):
    """
    A custom callback that logs information at the end of each episode.
    """
    def __init__(self, verbose: int = 0):
        super(EpisodeSummaryCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if the episode is done (dones is a list for vectorized envs)
        if self.locals['dones'][0]:
            # The 'info' dictionary contains the 'terminal_observation' when an episode ends,
            # captured by the Monitor wrapper.
            info = self.locals['infos'][0]
            episode = info.get('episode')
            terminal_obs = info.get('terminal_observation')
            #print(self.locals)
            #for kk, vv in self.model.get_parameters()['policy'].items():
            #    print(kk, vv.shape)
            #print("----------------------------------------------")

            if terminal_obs is not None:
                # The terminal_observation contains the final state before the reset.
                # obs format: [balance, crypto_holdings, price_window...]
                final_balance = terminal_obs[0]
                final_holdings = terminal_obs[1]
                # The last known price is the last element of the price window in the observation
                last_price = terminal_obs[-1]
                final_portfolio_value = final_balance + (final_holdings * last_price)

                # Get initial balance from the underlying environment
                initial_balance = self.training_env.get_attr('initial_balance')[0]

                print(f"\n--- Episode End Summary (Timestep: {self.num_timesteps}) ---")
                print(f"Lasted {episode['l']} timesteps")
                print(f"Collected rewards: {episode['r']}")
                print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
                print(f"Profit/Loss: {final_portfolio_value - initial_balance:.2f}")
                print(f"Final Balance: {final_balance:.2f}")
                print(f"Final Holdings: {final_holdings:.6f}")
        return True # Continue training

class StopTrainingOnEpisodeCount(BaseCallback):
    """
    Stops the training once a certain number of episodes have been completed.

    This is useful when episodes can have variable lengths.

    :param max_episodes: The number of episodes to train for.
    :param verbose: Verbosity level.
    """
    def __init__(self, max_episodes: int, verbose: int = 1):
        super(StopTrainingOnEpisodeCount, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # self.locals['dones'] is a boolean array, True for each env that is done.
        # In our case, we have a single environment, so we check dones[0].
        if self.locals['dones'][0]:
            self.episode_count += 1
            if self.verbose > 0:
                print(f"Episode {self.episode_count}/{self.max_episodes} finished.")

        if self.episode_count >= self.max_episodes:
            print(f"Stopping training: reached {self.episode_count} episodes.")
            return False  # Return False to stop training
        return True

if __name__ == '__main__':
    data_path = Path('data/')
    crypto_file = data_path / 'FORMEUR_1m_1095days.tsv'

    # Load the datasets
    print(f"Loading crypto data from: {crypto_file}")
    #crypto_prices_df = pd.read_csv(crypto_file, sep='\t') 
    crypto_prices_df_1min = generate_trending_sinusoid(num_days=355)

    # Subsample the data to an hourly frequency
    print(f"Original data points: {len(crypto_prices_df_1min)}")
    crypto_prices_df = subsample_data(crypto_prices_df_1min, freq='D')
    print(f"Subsampled data points (hourly): {len(crypto_prices_df)}")

    # 1. Create the base environment
    base_env = CryptoTraderEnv(
        initial_balance=1000,
        initial_crypto_holdings=0,
        crypto_prices=crypto_prices_df["close_eur"].values,
        window_size=30
    )
    # 2. Wrap it with a Monitor to automatically track episode stats (like terminal_observation)
    monitored_env = Monitor(base_env)
    # 3. Wrap it in a DummyVecEnv to make it compatible with SB3 algorithms
    env = DummyVecEnv([lambda: monitored_env])

    # For DQN, hyperparameters like buffer_size, train_freq, and target_update_interval
    # are passed during initialization, not during .learn()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=1e-3,
        batch_size=20,
        gradient_steps=1000,
        gamma=0.9999,
        train_freq=14,  # Update the model 
        target_update_interval=14, # Update the target network
        buffer_size=2000, # Size of the replay buffer
        learning_starts=14, # Number of steps to collect before training starts
        # --- Exploration parameters ---
        # The fraction of total training steps to anneal (decay) exploration rate
        #exploration_fraction=0.02,
        exploration_initial_eps=0.05, # Start with 100% random actions
        #exploration_final_eps=0.05,  # End with 5% random actions
    )

    # --- Training ---
    # Define the number of episodes you want to train for
    num_episodes_to_train = 1000

    # Create the callbacks
    episode_log_callback = EpisodeSummaryCallback()
    stop_on_episode_callback = StopTrainingOnEpisodeCount(max_episodes=num_episodes_to_train)
    callback_list = CallbackList([episode_log_callback, stop_on_episode_callback])

    print(f"Training for a maximum of {num_episodes_to_train} episodes.")
    model.learn(
        total_timesteps=float('inf'), # Train indefinitely until the callback stops it
        log_interval=1,
        progress_bar=False, # Disable progress bar as it's incompatible with infinite timesteps
        callback=callback_list,
    )
    model.save("dqn_crypto_trader")

    del model # remove to demonstrate saving and loading

    # When evaluating, we can use the base environment without the wrappers
    eval_env = base_env
    model = DQN.load("dqn_crypto_trader", env=eval_env)

    obs, info = eval_env.reset()
    # Run one full episode to evaluate the trained agent
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    # Render the environment to see what the agent is doing
    eval_env.render()

    print("\n--- Evaluation Finished ---")
    eval_env.close()