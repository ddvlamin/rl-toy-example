import gymnasium as gym
import numpy as np
import collections

class CryptoTraderEnv(gym.Env):
    """
    A cryptocurrency trading environment for reinforcement learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_balance=1000, initial_crypto_holdings=0, crypto_prices=None, transaction_fee_percent=0.001):
        super(CryptoTraderEnv, self).__init__()

        self.initial_balance = initial_balance
        self.initial_crypto_holdings = initial_crypto_holdings
        self.crypto_prices = crypto_prices if crypto_prices is not None else self._generate_dummy_prices()
        self.transaction_fee_percent = transaction_fee_percent

        # Internal state for tracking buy prices
        self._buys = collections.deque()

        self.current_step = 0

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: [balance, crypto_holdings, current_price]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.crypto_holdings = self.initial_crypto_holdings
        self.current_step = 0
        self._buys.clear()
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # If the episode is over, just return the final state
            return self._get_observation(), 0.0, self.done, False, {}

        current_price = self.crypto_prices[self.current_step]
        reward = 0.0

        if action == 1:  # Buy
            if self.balance > 0:
                # Spend 50% of balance to buy crypto
                spend_amount = self.balance * 0.5
                fee = spend_amount * self.transaction_fee_percent
                amount_to_buy = (spend_amount - fee) / current_price

                self.crypto_holdings += amount_to_buy
                self.balance -= spend_amount

                # Record the buy for reward calculation on sell
                self._buys.append({'amount': amount_to_buy, 'price': current_price})

                # Small penalty for transaction cost/discouraging frantic trading
                reward = -fee

        elif action == 2:  # Sell
            if self.crypto_holdings > 0:
                # Sell 50% of holdings
                crypto_to_sell = self.crypto_holdings * 0.5
                
                # Calculate the cost basis for the sold assets (FIFO - First-In, First-Out)
                cost_basis = 0
                amount_sold = crypto_to_sell
                while amount_sold > 0 and self._buys:
                    oldest_buy = self._buys[0]
                    if oldest_buy['amount'] <= amount_sold:
                        # This entire buy record is sold
                        cost_basis += oldest_buy['amount'] * oldest_buy['price']
                        amount_sold -= oldest_buy['amount']
                        self._buys.popleft()
                    else:
                        # Part of this buy record is sold
                        cost_basis += amount_sold * oldest_buy['price']
                        self._buys[0]['amount'] -= amount_sold
                        amount_sold = 0

                proceeds = crypto_to_sell * current_price
                fee = proceeds * self.transaction_fee_percent
                
                self.balance += proceeds - fee
                self.crypto_holdings -= crypto_to_sell

                # The primary reward: profit from the sale
                reward = (proceeds - fee) - cost_basis

        self.current_step += 1
        if self.current_step >= len(self.crypto_prices) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, False, {}

    def _get_observation(self):
        return np.array([
            self.balance,
            self.crypto_holdings,
            self.crypto_prices[self.current_step]
        ])

    def _generate_dummy_prices(self, n_steps=1000):
        # Generate a simple sine wave with some noise for dummy prices
        time = np.arange(0, n_steps)
        prices = 50000 + 10000 * np.sin(time * 0.1) + np.random.normal(0, 2000, n_steps)
        return prices

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
