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
                        "orig_balance\torig_holdings\torig_current_price\t" +
                        "orig_avg_cost\torig_avg_profit\torig_price_start\torig_price_trend\t" + 
                        "obs_balance\tobs_holdings\tobs_current_price\t" +
                        "obs_avg_cost\tobs_avg_profit\tobs_price_start\tobs_price_trend\n")
        self.episode_length = None

    def _on_training_start(self) -> None:
        # Cache static environment attributes once at the start to improve performance
        prices_len = self.training_env.get_attr('crypto_prices')[0].shape[0]
        window_size = self.training_env.get_attr('window_size')[0]
        self.episode_length = prices_len - window_size - 1

    def _on_step(self) -> bool:
        #get the action taken
        action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]

        done = self.locals['dones'][0]
        if done:
            info = self.locals['infos'][0]
            terminal_obs = info.get('terminal_observation')
            
            observation = terminal_obs
            if hasattr(self.training_env, 'unnormalize_obs'):
                original_observation = self.training_env.unnormalize_obs(terminal_obs)
            else:
                original_observation = terminal_obs
            
            step = self.training_env.get_attr('crypto_prices')[0].shape[0] - 1
            episode = (self.num_timesteps - 1) // self.episode_length
        else:
            observation = self.locals['new_obs'][0]
            original_observation = self.training_env.get_original_obs()[0]
            step = self.training_env.get_attr('current_step')[0]
            episode = self.num_timesteps // self.episode_length

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
                           "\t".join([f"{x:.6f}" for x in original_observation]) + "\t" +
                           "\t".join([f"{x:.6f}" for x in observation]) + "\n")

        return True # Continue training

    def _on_training_end(self) -> None:
        print("training ended, closing logfile")
        self.logfile.close()

    def close(self):
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
                
                # Unnormalize the terminal observation to get real values
                if hasattr(self.training_env, 'unnormalize_obs'):
                    original_terminal_obs = self.training_env.unnormalize_obs(terminal_obs)
                else:
                    original_terminal_obs = terminal_obs
                
                # Calculate final asset value from the terminal state
                final_asset_value = original_terminal_obs[0] + original_terminal_obs[1] * original_terminal_obs[2]

                print(f"\n--- Episode End Summary (Timestep: {self.num_timesteps}) ---")
                print(f"Lasted {episode['l']} timesteps")
                print(f"Collected rewards: {episode['r']}")
                print(f"Initial balance: {initial_balance:.2f}")
                print(f"Last asset value: {final_asset_value:.2f}")
                print(f"Profit/Loss: {final_asset_value - initial_balance:.2f}")
                print(f"Last Balance: {original_terminal_obs[0]:.2f}")
                print(f"Last Holdings: {original_terminal_obs[1]:.6f}")
                print(f"Last Price: {original_terminal_obs[2]:.2f}")
                print(f"Last Avg Cost: {original_terminal_obs[3]:.2f}")
                print(f"Last Avg Profit:   {original_terminal_obs[4]:.2f}")
                print(f"Last Price Start:  {original_terminal_obs[5]:.2f}")
                print(f"Last Price Trend:  {original_terminal_obs[6]:.4f}")


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