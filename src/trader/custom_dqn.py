
from stable_baselines3 import DQN
import torch as th
from torch.nn import functional as F
import numpy as np

class LoggingDQN(DQN):
    """
    Custom DQN class that logs input, output, and target values during training.
    """
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        # Lists to store the stats for logging
        input_means = []
        output_means = []
        target_means = []
        next_q_means = []

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow Greedy Policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                next_q_means.append(next_q_values.mean().item())
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (smooth L1 loss)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            
            # --- Capture Stats ---
            # Log the mean of the input observations (normalized if VecNormalize is used)
            input_means.append(replay_data.observations.mean().item())
            # Log the mean of the predicted Q-values
            output_means.append(current_q_values.mean().item())
            # Log the mean of the target Q-values
            target_means.append(target_q_values.mean().item())
            # ---------------------

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        
        # --- Log Stats ---
        self.logger.record("train/mean_input_observation", np.mean(input_means))
        self.logger.record("train/mean_output_q_value", np.mean(output_means))
        self.logger.record("train/mean_target_q_value", np.mean(target_means))
        self.logger.record("train/mean_next_q_value", np.mean(next_q_means))
