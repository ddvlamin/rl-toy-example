import gymnasium as gym
import numpy as np
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

    def __init__(self, initial_balance=1000, initial_crypto_holdings=0, crypto_prices=None, transaction_fee_percent=0.01, window_size=10):
        super(CryptoTraderEnv, self).__init__()

        self.initial_balance = initial_balance
        self.initial_crypto_holdings = initial_crypto_holdings
        self.crypto_prices = crypto_prices
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size


        # Internal state for tracking average cost basis
        self.total_cost_basis = 0.0

        self.current_step = 0

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(len(Actions))

        # Observation space: [balance, crypto_holdings, current_price]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.window_size+2,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.crypto_holdings = self.initial_crypto_holdings
        self.current_step = self.window_size
        self.total_cost_basis = 0.0
        self.done = False
        self.last_portfolio_value = self._get_assets_value()
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # If the episode is over, just return the final state
            return self._get_observation(), 0.0, self.done, False, {}

        current_price = self.crypto_prices[self.current_step]
        current_portfolio_value = self._get_assets_value()
        reward = 0.0

        if action == Actions.Buy:
            if self.balance > 1:
                # Spend 50% of balance to buy crypto
                spend_amount = self.balance * 0.5
                # The fee is part of the spend, not extra
                fee = spend_amount * self.transaction_fee_percent
                net_spend = spend_amount - fee
                amount_to_buy = net_spend / current_price

                self.crypto_holdings += amount_to_buy
                self.balance -= spend_amount
                self.total_cost_basis += net_spend

                #reward = current_portfolio_value - self.last_portfolio_value

                # Small penalty for transaction cost/discouraging frantic trading
                reward -= fee

            # if balance is very low and crypto is worth nothing, end the episode
            elif self.crypto_holdings * current_price < 1:
                reward = self.balance - self.initial_balance
                self.done = True
            
            # Not enough balance to buy
            else:
                reward = -1.0  # Small penalty for trying to buy without funds

        elif action == Actions.Sell:
            if self.crypto_holdings * current_price > 1:
                # Sell 50% of holdings
                crypto_to_sell = self.crypto_holdings * 0.5

                proceeds = crypto_to_sell * current_price
                fee = proceeds * self.transaction_fee_percent
                net_proceeds = proceeds - fee
                
                # Calculate cost basis of the sold portion using average cost
                proportion_sold = crypto_to_sell / self.crypto_holdings
                cost_basis_of_sale = self.total_cost_basis * proportion_sold

                # The primary reward: profit from the sale
                reward = net_proceeds - cost_basis_of_sale

                # Update state
                self.balance += net_proceeds
                self.crypto_holdings -= crypto_to_sell
                self.total_cost_basis -= cost_basis_of_sale

            # if crypto is worth nothing and the balance is very low, end the episode
            elif self.balance < 1: # Reset if holdings are negligible
                reward = self.balance - self.initial_balance
                self.done = True
            
            # Not enough crypto to sell
            else:
                reward = -1.0  # Small penalty for trying to sell without holdings

        elif action == Actions.Hold:
            pass

        self.last_portfolio_value = current_portfolio_value
        self.current_step += 1
        if self.current_step >= len(self.crypto_prices) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, False, {}

    def _get_observation(self):
        price_window = self.crypto_prices[self.current_step - self.window_size : self.current_step]
        return np.concatenate((
            [self.balance, self.crypto_holdings],
            price_window
        ))
    
    def _get_assets_value(self):
        current_price = self.crypto_prices[self.current_step]
        return self.balance + self.crypto_holdings * current_price

    def render(self, mode='human', close=False):
        current_price = self.crypto_prices[self.current_step]
        portfolio_value = self.balance + self.crypto_holdings * current_price
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
