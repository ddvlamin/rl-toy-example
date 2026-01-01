import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    """
    A custom callback that logs information at the end of each episode.
    """
    def __init__(self, logfile, verbose: int = 0):
        super(EpisodeLogger, self).__init__(verbose)
        self.logfile = open(logfile, 'w')
        #write header
        self.logfile.write("episode\tstep\taction\treward\t" +
                           "obs_balance\tobs_holdings\tobs_current_price\t" +
                           "obs_avg_cost\tobs_avg_profit\tobs_price_start\tobs_price_trend\t" +
                           "orig_balance\torig_holdings\torig_current_price\t" +
                           "orig_avg_cost\torig_avg_profit\torig_price_start\torig_price_trend\n")

    def _on_step(self) -> bool:
        #get the action taken
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        episode_length = self.training_env.get_attr('crypto_prices')[0].shape[0] - self.training_env.get_attr('window_size')[0] - 1

        done = self.locals['dones'][0]
        if done:
            info = self.locals['infos'][0]
            terminal_obs = info.get('terminal_observation')
            # Unnormalize if possible to match the raw logging format
            if hasattr(self.training_env, 'unnormalize_obs'):
                observation = self.training_env.unnormalize_obs(terminal_obs)
            else:
                observation = terminal_obs
            original_observation = observation
            step = self.training_env.get_attr('crypto_prices')[0].shape[0] - 1
            episode = (self.num_timesteps - 1) // episode_length
        else:
            observation = self.training_env.envs[0].env._get_observation()
            original_observation = self.training_env.get_original_obs()[0]
            step = self.training_env.get_attr('current_step')[0]
            episode = self.num_timesteps // episode_length

        # Get the episode rewards from the Monitor wrapper
        episode_rewards = self.training_env.env_method('get_episode_rewards')
        #print(episode_rewards)
        if episode_rewards:
            # The result is a list of deques, one for each env. We have one env.
            rewards_deque = episode_rewards[0]
            if rewards_deque:
                ep_rew_mean = np.mean(rewards_deque)
                #print(f"Mean reward over last {len(rewards_deque)} episodes: {ep_rew_mean:.2f}")        

        #write to logfile as tsv
        self.logfile.write(f"{episode}\t{step}\t{action}\t{reward}\t" +
                           "\t".join([f"{x:.6f}" for x in observation]) + "\t" +
                           "\t".join([f"{x:.6f}" for x in original_observation]) + "\n")

        return True # Continue training

    def _on_training_end(self) -> None:
        print("training ended, closing logfile")
        self.logfile.close()

    def close():
        self.logfile.close()



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

             # Update any final state if needed
            #print(self.locals)
            #for kk, vv in self.model.get_parameters()['policy'].items():
            #    print(kk, vv.shape)
            #print("----------------------------------------------")

            if terminal_obs is not None:
                # Get initial balance from the underlying environment
                initial_balance = self.training_env.get_attr('initial_balance')[0]
                crypto_prices = self.training_env.get_attr('crypto_prices')[0]
                # Get the original, unnormalized observation from the VecNormalize wrapper
                original_terminal_obs = self.training_env.get_original_obs()
                # The observation from a VecEnv is a batch of 1. We get the first element.
                original_terminal_obs = original_terminal_obs[0]
                last_price = crypto_prices[self.last_step]

                print(f"\n--- Episode End Summary (Timestep: {self.num_timesteps}) ---")
                print(f"Lasted {episode['l']} timesteps")
                print(f"Last step {self.last_step}")
                print(f"Collected rewards: {episode['r']}")
                print(f"Initial balance: {initial_balance:.2f}")
                print(f"Last asset value: {self.last_asset_value:.2f}")
                print(f"Profit/Loss: {self.last_asset_value - initial_balance:.2f}")
                print(f"Last Balance: {self.last_balance:.2f}")
                print(f"Last Holdings: {self.last_crypto_holdings:.6f}")
                print(f"Last price: {last_price:.6f}")
                print("---")
                print(f"Normalized Terminal Obs: {terminal_obs}")
                print(f"Original Terminal Obs:   {original_terminal_obs}")
                print(f"  -> Balance:      {original_terminal_obs[0]:.2f}")
                print(f"  -> Holdings:     {original_terminal_obs[1]:.6f}")
                print(f"  -> Current Price:  {original_terminal_obs[2]:.2f}")
                print(f"  -> Avg Cost:     {original_terminal_obs[3]:.2f}")
                print(f"  -> Avg Profit:   {original_terminal_obs[4]:.2f}")
                print(f"  -> Price Start:  {original_terminal_obs[5]:.2f}")
                print(f"  -> Price Trend:  {original_terminal_obs[6]:.4f}")
        else:
            self.last_asset_value = self.training_env.envs[0].env._get_assets_value()
            self.last_crypto_holdings = self.training_env.get_attr('crypto_holdings')[0]
            self.last_balance = self.training_env.get_attr('balance')[0]
            self.last_step = self.training_env.get_attr('current_step')[0]

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