from stable_baselines3.common.callbacks import BaseCallback


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