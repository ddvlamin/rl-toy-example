import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    """
    A custom callback that logs information for each step
    """
    def __init__(self, logfile, verbose: int = 0):
        super(EpisodeLogger, self).__init__(verbose)
        self.logfile = open(logfile, 'w')
        # write header
        self.logfile.write("episode\tstep\taction\treward\t" +
                        "policy_q\ttarget_q\t" +
                        "orig_log_return\torig_balance_ratio\torig_holdings_ratio\t" +
                        "orig_price_trend\torig_profit_ratio\torig_price_diff_last_action\torig_portfolio_ratio\t" + 
                        "obs_log_return\tobs_balance_ratio\tobs_holdings_ratio\t" +
                        "obs_price_trend\tobs_profit_ratio\tobs_price_diff_last_action\tobs_portfolio_ratio\n")
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

        # Calculate Q-values
        obs = self.model._last_obs
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with th.no_grad():
            q_values = self.model.q_net(obs_tensor)
            q_values_target = self.model.q_net_target(obs_tensor)
        
        policy_q = q_values[0, int(action)].item()
        target_q = q_values_target[0, int(action)].item()

        #write to logfile as tsv
        self.logfile.write(f"{episode}\t{step}\t{action}\t{reward}\t" +
                           f"{policy_q:.6f}\t{target_q:.6f}\t" +
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
            info = self.locals['infos'][0]
            episode = info.get('episode')
            
            # Get values directly from the environment
            initial_balance = self.training_env.get_attr('initial_balance')[0]
            balance = self.training_env.get_attr('balance')[0]
            crypto_holdings = self.training_env.get_attr('crypto_holdings')[0]
            current_step = self.training_env.get_attr('current_step')[0]
            crypto_prices = self.training_env.get_attr('crypto_prices')[0]
            current_price = crypto_prices[current_step]
            avg_cost = self.training_env.get_attr('avg_cost_of_holdings')[0]
            
            final_asset_value = balance + crypto_holdings * current_price

            print(f"\n--- Episode End Summary (Timestep: {self.num_timesteps}) ---")
            print(f"Lasted {episode['l']} timesteps")
            print(f"Collected rewards: {episode['r']}")
            print(f"Initial balance: {initial_balance:.2f}")
            print(f"Last asset value: {final_asset_value:.2f}")
            print(f"Profit/Loss: {final_asset_value - initial_balance:.2f}")
            print(f"Last Balance: {balance:.2f}")
            print(f"Last Holdings: {crypto_holdings:.6f}")
            print(f"Last Price: {current_price:.2f}")
            print(f"Last Avg Cost: {avg_cost:.2f}")

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