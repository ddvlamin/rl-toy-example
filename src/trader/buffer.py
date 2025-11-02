import numpy as np
import torch
from typing import Optional
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from stable_baselines3.common.vec_env import VecNormalize

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, top_priority_count=10, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self.model = None  # Will be set after model initialization
        self.top_priority_count = top_priority_count
        self.worst_samples = []
        self.best_samples = []
        print(f"Initialized PrioritizedReplayBuffer with top_priority_count={top_priority_count}")

    def add(self, obs, next_obs, action, reward, done, infos):
        reward_target = reward
        # Calculate discounted future reward (TD Target) if the model is set
        if self.model is not None and self.model.q_net_target is not None:
            with torch.no_grad():
                # We don't want to backprop through the target network
                next_obs_tensor = self.to_torch(next_obs).to(self.device)
                # Compute the Q-values for the next observation
                next_q_values = self.model.q_net_target(next_obs_tensor)
                # The target for the Q-value is the reward + discounted max Q-value of the next state
                reward_target = reward + (1 - done) * self.model.gamma * next_q_values.max().item()
                reward_target = float(reward_target[0])

        # Store the raw, unnormalized data. Normalization will happen at sample time.
        data = (obs,
                next_obs,
                action,
                reward, # In this custom buffer, we are storing the raw reward
                done,
                infos)
        if len(self.best_samples) < self.top_priority_count or reward_target > self.best_samples[0][0]:
            self.best_samples.append((reward_target, data))
            self.best_samples.sort(reverse=True, key = lambda x: x[0])
            self.best_samples = self.best_samples[:self.top_priority_count]

        if len(self.worst_samples) < self.top_priority_count or reward_target < self.worst_samples[0][0]:
            self.worst_samples.append((reward_target, data))
            self.worst_samples.sort(key = lambda x: x[0])
            self.worst_samples = self.worst_samples[:self.top_priority_count]

        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env: Optional[VecNormalize] = None):
        # first sample from the prioritized buffers
        num_prioritized = min(batch_size // 2, len(self.best_samples) + len(self.worst_samples))

        best_indices = np.random.choice(len(self.best_samples), size=num_prioritized // 2, replace=len(self.best_samples) < num_prioritized // 2)
        worst_indices = np.random.choice(len(self.worst_samples), size=num_prioritized - num_prioritized // 2, replace=len(self.worst_samples) < num_prioritized - num_prioritized // 2)

        prioritized_samples = [self.best_samples[i][1] for i in best_indices] + [self.worst_samples[i][1] for i in worst_indices]

        # then sample from the normal buffer to fill out the rest of the batch
        remaining_batch = batch_size - num_prioritized

        # Get the regular samples from the base buffer
        regular_samples = super().sample(remaining_batch, env)

        # Convert our list of tuples into a ReplayBufferSamples object
        # This is what the base class's _get_samples method does.
        obs_, actions_, next_obs_, dones_, rewards_ = self.__get_samples(prioritized_samples, env)

        # Concatenate the tensors from both sample sets
        combined_obs = torch.cat((regular_samples.observations, obs_), dim=0)
        combined_actions = torch.cat((regular_samples.actions, actions_), dim=0)
        combined_next_obs = torch.cat((regular_samples.next_observations, next_obs_), dim=0)
        combined_dones = torch.cat((regular_samples.dones, dones_), dim=0)
        combined_rewards = torch.cat((regular_samples.rewards, rewards_), dim=0)

        replay_data = (
            combined_obs,
            combined_actions,
            combined_next_obs,
            combined_dones,
            combined_rewards,
        )
    
        return ReplayBufferSamples(*tuple(map(self.to_torch, replay_data)))

    def __get_samples(self, samples_list, env: Optional[VecNormalize] = None):
        # This helper function mimics parts of the base class's _get_samples
        # to convert our list of tuples into the correct tensor format.
        obs, next_obs, actions, rewards, dones = zip(*[s[:5] for s in samples_list])

        # Convert to numpy arrays
        obs = np.array(obs).reshape(len(samples_list), -1)
        next_obs = np.array(next_obs).reshape(len(samples_list), -1)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Normalize the observations using the running stats from VecNormalize
        obs = self._normalize_obs(obs, env)
        next_obs = self._normalize_obs(next_obs, env)
        rewards = self._normalize_reward(rewards, env)


        # Convert to PyTorch tensors
        obs_tensor = self.to_torch(obs)
        next_obs_tensor = self.to_torch(next_obs)
        actions_tensor = self.to_torch(actions).reshape(-1, 1)
        rewards_tensor = self.to_torch(rewards).reshape(-1, 1)
        dones_tensor = self.to_torch(dones).reshape(-1, 1)

        return (
            obs_tensor,
            actions_tensor,
            next_obs_tensor,
            dones_tensor,
            rewards_tensor
        )

        