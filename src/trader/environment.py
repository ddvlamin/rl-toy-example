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

        # Observation space: [log_return, balance_ratio, holdings_ratio, price_trend_normalized]
        # All features are now relative or normalized.
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
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
        self.last_action_price = self.crypto_prices[self.current_step]
        return self._get_observation(), {}

    def step(self, action):
        # 1. Execute the action
        penalty = self._execute_trade(action)
        
        # 2. Update the step and check for episode end
        self.current_step += 1
        if self.current_step >= len(self.crypto_prices) - 1:
            self.done = True
            
        # 3. Calculate reward based on portfolio value change
        current_portfolio_value = self._get_assets_value()
        # Using log-return as a stationary reward signal
        reward = math.log(current_portfolio_value / self.last_portfolio_value)
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
                self.last_action_price = current_price
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
                self.last_action_price = current_price
                self.avg_cost_of_holdings = 0.0
                return 0.0
            else:
                return -self.invalid_action_penalty

        elif action == Actions.Hold:
            return 0.0

    def _get_observation(self):
        # We want features that are stationary (don't grow indefinitely)
        current_price = self.crypto_prices[self.current_step]
        price_start = self.crypto_prices[self.current_step - self.window_size]
        
        # 1. Log-return of the current price compared to the window start
        log_return = math.log(current_price / price_start)
        
        # 2. Relative balance (balance / total_assets)
        total_assets = self._get_assets_value()
        balance_ratio = self.balance / total_assets
        
        # 3. Relative holdings (holdings_value / total_assets)
        holdings_ratio = (self.crypto_holdings * current_price) / total_assets
        
        # 4. Price trend relative to initial price of the window
        price_trend = (current_price - price_start) / price_start
        
        # 5. Position relative to average cost (if any)
        if self.crypto_holdings > 0:
            profit_ratio = (current_price - self.avg_cost_of_holdings) / self.avg_cost_of_holdings
        else:
            profit_ratio = 0.0
            
        # 6. Distance from last action price
        price_diff_last_action = (current_price - self.last_action_price) / self.last_action_price
        
        # 7. Portfolio value ratio (relative to initial) - but this might be non-stationary.
        # Let's use log(total_assets / initial_balance) instead.
        portfolio_ratio = math.log(total_assets / self.initial_balance)

        return np.array([
            log_return,
            balance_ratio,
            holdings_ratio,
            price_trend,
            profit_ratio,
            price_diff_last_action,
            portfolio_ratio
        ], dtype=np.float32)
    
    def _get_assets_value(self):
        current_price = self.crypto_prices[self.current_step]
        return self.balance + self.crypto_holdings * current_price

    def get_original_obs(self):
        """
        Returns the unnormalized observation. 
        Used by callbacks for logging purposes.
        """
        return self._get_observation()

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
    # Create dummy price data for the example
    dummy_prices = np.linspace(100, 110, 100) + np.random.normal(0, 1, 100)
    env = CryptoTraderEnv(crypto_prices=dummy_prices, window_size=10)
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")
