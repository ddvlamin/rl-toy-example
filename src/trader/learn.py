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
    print(f"Subsampled data points (hourly): {len(crypto_prices_df)}")

    feature_window_size = 30
    # The length of one episode is the number of prices minus the window size.
    episode_length = len(crypto_prices_df) - feature_window_size - 1

    # 1. Create the base environment
    base_env = CryptoTraderEnv(
        initial_balance=1000,
        crypto_prices=crypto_prices_df["close_eur"].values,
        window_size=feature_window_size
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
        net_arch=[32, 32, 32],  # Two hidden layers with 32 neurons each
    )
    model = LoggingDQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_rate=1e-3,
        batch_size=40,
        #replay_buffer_class=PrioritizedReplayBuffer,
        #replay_buffer_kwargs=dict(top_priority_count=int(episode_length*100/2)),
        gradient_steps=1000,
        gamma=0.9999,
        train_freq=episode_length,  # Update the model every episode
        target_update_interval=episode_length*10, # Update the target network every 10 episodes
        buffer_size=episode_length*100, # 100 episodes
        learning_starts=episode_length, # Number of steps to collect before training starts
        # --- Exploration parameters ---
        # The fraction of total training steps to anneal (decay) exploration rate
        exploration_fraction=0.2,
        exploration_initial_eps=0.3, # Start with 100% random actions
        exploration_final_eps=0.05,  # End with 5% random actions
    )
    model.set_logger(new_logger)

    # Set the model in the custom replay buffer after model initialization
    #model.replay_buffer.model = model

    # --- Training ---
    # Define the number of episodes to train for
    num_episodes_to_train = 5000
    # Calculate the approximate total timesteps. This is needed for the exploration decay schedule.
    total_timesteps_estimate = num_episodes_to_train * episode_length

    # Create the callbacks
    episode_log_callback = EpisodeSummaryCallback()
    episode_logger = EpisodeLogger(logfile=str(log_dir / "episode_steps_log.tsv"))
    # The StopTrainingOnEpisodeCount callback is no longer needed as we provide a finite total_timesteps.
    callback_list = CallbackList([episode_logger,episode_log_callback])

    print(f"Training for {num_episodes_to_train} episodes (approx. {total_timesteps_estimate} timesteps).")
    model.learn(
        total_timesteps=total_timesteps_estimate,
        log_interval=1,
        callback=callback_list,
    )

    # --- Saving ---
    # Save the model
    model.save("dqn_crypto_trader")
    # IMPORTANT: Save the VecNormalize statistics
    env.save("vec_normalize_stats.pkl")
    print("Training complete. Model saved as 'dqn_crypto_trader'.")