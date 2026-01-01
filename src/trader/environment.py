import gymnasium as gym
import numpy as np
import math
from enum import IntEnum

class Actions(IntEnum):
    Hold = 0
    Buy = 1
    Sell = 2

class CryptoTraderEnv(gym.Env):
    """
    A cryptocurrency trading environment for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_balance=1000, crypto_prices=None, transaction_fee_percent=0.01, window_size=10, invalid_action_penalty=0.1):
        super(CryptoTraderEnv, self).__init__()

        self.initial_balance = initial_balance
        self.crypto_prices = crypto_prices
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.invalid_action_penalty = invalid_action_penalty

        self.current_step = self.window_size

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(len(Actions))

        # Observation space: [balance, crypto_holdings, current_price]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf, # can be np.inf as values are normalized
            shape=(7,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.crypto_holdings = 0.0
        self.current_step = self.window_size
        self.done = False
        self.last_portfolio_value = self.initial_balance
        self.avg_cost_of_holdings = 0.0
        self.avg_profit = self.crypto_prices[0]
        return self._get_observation(), {}

    def step(self, action):
        #if self.done:
            # If the episode is over, just return the final state
        #    return self._get_observation(), 0.0, self.done, False, {}

        # 1. Execute the action
        penalty = self._execute_trade(action)
        
        # 2. Update the step and check for episode end
        self.current_step += 1
        if self.current_step >= len(self.crypto_prices) - 1:
            self.done = True
            
        # 3. Calculate reward based on portfolio value change
        current_portfolio_value = self._get_assets_value()
        reward = math.log(current_portfolio_value / self.last_portfolio_value) # Log-return reward
        reward += penalty
        self.last_portfolio_value = current_portfolio_value
        
        # 4. Get the next observation
        obs = self._get_observation()
        
        return obs, reward, self.done, False, { "is_success": current_portfolio_value > self.initial_balance }
    
    def _execute_trade(self, action):
        current_price = self.crypto_prices[self.current_step]

        if action == Actions.Buy:
            # Buy with all available balance
            if self.balance > 0:
                fee = self.transaction_fee_percent * self.balance
                buy_amount = self.balance - fee
                self.crypto_holdings += buy_amount / current_price
                self.balance = 0
                self.avg_profit = 0.0
                self.avg_cost_of_holdings = current_price
                return 0.0
            else:
                return -self.invalid_action_penalty

        elif action == Actions.Sell:
            # Sell all available holdings
            if self.crypto_holdings > 0:
                sell_amount = self.crypto_holdings * current_price
                fee = self.transaction_fee_percent * sell_amount
                sell_amount -= fee
                self.balance += sell_amount
                self.crypto_holdings = 0
                self.avg_cost_of_holdings = 0.0
                self.avg_profit = current_price
                return 0.0
            else:
                return -self.invalid_action_penalty

        elif action == Actions.Hold:
            pass

    def _get_observation(self):
        # Return raw, unnormalized values. VecNormalize will handle normalization.
        price_start = self.crypto_prices[self.current_step - self.window_size]
        current_price = self.crypto_prices[self.current_step]
        price_trend = (current_price - price_start) / self.window_size

        return np.concatenate((
            [
                self.balance,
                self.crypto_holdings,
                current_price,
                self.avg_cost_of_holdings,
                self.avg_profit,
                price_start,
                price_trend,
            ],
        ))
    
    def _get_assets_value(self):
        current_price = self.crypto_prices[self.current_step]
        return self.balance + self.crypto_holdings * current_price

    def render(self, mode='human', close=False):
        current_price = self.crypto_prices[self.current_step]
        portfolio_value = self._get_assets_value()
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Crypto Holdings: {self.crypto_holdings:.6f}")
        print(f"Current Price: {current_price:.2f}")
        print(f"Portfolio Value: {portfolio_value:.2f}")

if __name__ == '__main__':
    # Example of how to use the environment
    env = CryptoTraderEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")
