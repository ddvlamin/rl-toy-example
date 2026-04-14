from pathlib import Path
import os

from stable_baselines3 import DQN
from stable_baselines3 import DQN, common

from trader.environment import CryptoTraderEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

from trader.sinusoid import generate_trending_sinusoid
from trader.signals import subsample_data
#from trader.buffer import PrioritizedReplayBuffer

from trader.custom_dqn import LoggingDQN
from trader.callbacks import EpisodeSummaryCallback, StopTrainingOnEpisodeCount, EpisodeLogger


if __name__ == '__main__':
    data_path = Path('data/')
    crypto_file = data_path / 'FORMEUR_1m_1095days.tsv'
    log_dir = Path("logs/")
    os.makedirs(log_dir, exist_ok=True)

    new_logger = configure(str(log_dir), ["stdout", "csv"])

    # Load the datasets
    print(f"Loading crypto data from: {crypto_file}")
    #crypto_prices_df = pd.read_csv(crypto_file, sep='\t') 
    crypto_prices_df_1min = generate_trending_sinusoid(num_days=355)

    # Subsample the data to an hourly frequency
    print(f"Original data points: {len(crypto_prices_df_1min)}")
    crypto_prices_df = subsample_data(crypto_prices_df_1min, freq='D')
    print(f"Subsampled data points (daily): {len(crypto_prices_df)}")

    feature_window_size = 30
    # The length of one episode is the number of prices minus the window size.
    episode_length = len(crypto_prices_df) - feature_window_size - 1

    # 1. Create the base environment
    base_env = CryptoTraderEnv(
        initial_balance=1000,
        crypto_prices=crypto_prices_df["close_eur"].values,
        window_size=feature_window_size,
        invalid_action_penalty=0.01 # Reduced penalty
    )
    # 2. Wrap it with a Monitor to automatically track episode stats (like terminal_observation)
    monitored_env = Monitor(base_env, str(log_dir), info_keywords=("is_success",))
    # 3. Wrap it in a DummyVecEnv to make it compatible with SB3 algorithms
    vec_env = DummyVecEnv([lambda: monitored_env])
    # 4. Wrap it with VecNormalize to normalize observations
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)


    # For DQN, hyperparameters like buffer_size, train_freq, and target_update_interval
    # are passed during initialization, not during .learn()
    policy_kwargs = dict(
        net_arch=[64, 64],  # Slightly larger network
    )
    model = LoggingDQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4, # Standard learning rate for Adam
        batch_size=64,
        gradient_steps=1, # Update once per step
        gamma=0.99, # Standard discount factor
        train_freq=4,  # Update the model every 4 steps
        target_update_interval=1000, # Update the target network every 1000 steps
        buffer_size=100000,
        learning_starts=1000, 
        # --- Exploration parameters ---
        exploration_fraction=0.5, # Longer annealing
        exploration_initial_eps=1.0, 
        exploration_final_eps=0.05,
    )
    model.set_logger(new_logger)

    # --- Training ---
    num_episodes_to_train = 5000
    total_timesteps_estimate = num_episodes_to_train * episode_length

    # Create the callbacks
    episode_log_callback = EpisodeSummaryCallback()
    episode_logger = EpisodeLogger(logfile=str(log_dir / "episode_steps_log.tsv"))
    callback_list = CallbackList([episode_logger,episode_log_callback])

    print(f"Training for {num_episodes_to_train} episodes (approx. {total_timesteps_estimate} timesteps).")
    model.learn(
        total_timesteps=total_timesteps_estimate,
        log_interval=10,
        callback=callback_list,
    )

    # --- Saving ---
    model.save("dqn_crypto_trader")
    env.save("vec_normalize_stats.pkl")
    print("Training complete. Model saved as 'dqn_crypto_trader'.")