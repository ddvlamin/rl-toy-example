"""Custom Stable-Baselines3 callbacks for step and episode logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TextIO

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class EpisodeLogger(BaseCallback):
    """Callback that logs step-level rewards, Q-values, and observations to TSV."""

    def __init__(self, logfile: str | Path, verbose: int = 0) -> None:
        """Initialize the step logger callback.

        Args:
            logfile: Destination path for the tab-separated step log.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.logfile_path: Path = Path(logfile)
        self.logfile: TextIO | None = None
        self.episode_length: int = 1

    def _on_training_start(self) -> None:
        """Open logfile lazily, cache episode length, and write feature header."""
        self.episode_length = int(self.training_env.get_attr("episode_length")[0])
        feature_names: list[str] = self.training_env.get_attr("feature_names")[0]
        if self.logfile is None or self.logfile.closed:
            self.logfile_path.parent.mkdir(parents=True, exist_ok=True)
            self.logfile = self.logfile_path.open("w", encoding="utf-8")
            orig_cols = "\t".join(f"orig_{name}" for name in feature_names)
            obs_cols = "\t".join(f"obs_{name}" for name in feature_names)
            self.logfile.write(
                "episode\tstep\taction\treward\tpolicy_q\ttarget_q\t"
                f"{orig_cols}\t{obs_cols}\n"
            )

    def _on_step(self) -> bool:
        """Record transition metrics and observations at each environment step."""
        if self.logfile is None or self.logfile.closed:
            raise RuntimeError(
                "EpisodeLogger logfile is not open; ensure _on_training_start ran."
            )

        action = int(self.locals["actions"][0])
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        if done:
            info = self.locals["infos"][0]
            terminal_obs = info.get("terminal_observation")
            if terminal_obs is None:
                raise RuntimeError(
                    "Missing terminal_observation in terminal step info."
                )
            observation: np.ndarray = terminal_obs
            if hasattr(self.training_env, "unnormalize_obs"):
                original_observation = self.training_env.unnormalize_obs(terminal_obs)
            else:
                original_observation = terminal_obs
            step = int(
                info.get(
                    "step",
                    self.training_env.get_attr("crypto_prices")[0].shape[0] - 1,
                )
            )
            episode = (self.num_timesteps - 1) // self.episode_length
        else:
            observation = self.locals["new_obs"][0]
            original_observation = self.training_env.env_method("get_original_obs")[0]
            step = int(self.training_env.get_attr("current_step")[0])
            episode = self.num_timesteps // self.episode_length

        policy_q, target_q = self._compute_q_values(action)

        orig_str = "\t".join(f"{float(x):.6f}" for x in original_observation)
        obs_str = "\t".join(f"{float(x):.6f}" for x in observation)
        self.logfile.write(
            f"{episode}\t{step}\t{action}\t{reward}\t"
            f"{policy_q:.6f}\t{target_q:.6f}\t{orig_str}\t{obs_str}\n"
        )
        return True

    def _compute_q_values(self, action: int) -> tuple[float, float]:
        """Compute policy/target Q-values (DQN) or critic state value (Actor-Critic)."""
        if isinstance(self.model, DQN) and self.model._last_obs is not None:
            obs_tensor, _ = self.model.policy.obs_to_tensor(self.model._last_obs)
            with th.no_grad():
                q_values = self.model.q_net(obs_tensor)
                q_values_target = self.model.q_net_target(obs_tensor)
            return float(q_values[0, action].item()), float(
                q_values_target[0, action].item()
            )
        values = self.locals.get("values")
        if isinstance(values, th.Tensor):
            return float(values.flatten()[0].item()), 0.0
        return 0.0, 0.0

    def _on_training_end(self) -> None:
        """Close the TSV logfile when training completes."""
        self.close()

    def close(self) -> None:
        """Flush and close the underlying logfile if still open."""
        if self.logfile is not None and not self.logfile.closed:
            self.logfile.close()


class EpisodeSummaryCallback(BaseCallback):
    """Callback that logs portfolio summary metrics at the end of each episode."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize the episode summary callback.

        Args:
            verbose: Verbosity level.
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Log terminal summary statistics preserved in step info before auto-reset."""
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            episode = info.get("episode")
            initial_balance = float(self.training_env.get_attr("initial_balance")[0])

            final_asset_value = float(info.get("portfolio_value", initial_balance))
            balance = float(info.get("balance", initial_balance))
            crypto_holdings = float(info.get("crypto_holdings", 0.0))
            current_price = float(info.get("current_price", 0.0))
            raw_entry = info.get("entry_price")
            entry_price = 0.0 if raw_entry is None else float(raw_entry)

            if self.verbose > 0 and episode is not None:
                logger.info(
                    "Episode End (t=%d) | len=%s | rew=%s | value=%.2f | PnL=%.2f | "
                    "balance=%.2f | holdings=%.6f | price=%.2f | entry_price=%.2f",
                    self.num_timesteps,
                    episode["l"],
                    episode["r"],
                    final_asset_value,
                    final_asset_value - initial_balance,
                    balance,
                    crypto_holdings,
                    current_price,
                    entry_price,
                )
        return True


class StopTrainingOnEpisodeCount(BaseCallback):
    """Callback that stops training once a target episode count is reached."""

    def __init__(self, max_episodes: int, verbose: int = 1) -> None:
        """Initialize the episode-count stopping criterion.

        Args:
            max_episodes: Maximum number of completed episodes before stopping.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.max_episodes: int = max_episodes
        self.episode_count: int = 0

    def _on_step(self) -> bool:
        """Increment episode counter on termination and stop when limit reached."""
        if self.locals["dones"][0]:
            self.episode_count += 1
            if self.verbose > 0:
                logger.info(
                    "Episode %d/%d finished.", self.episode_count, self.max_episodes
                )

        if self.episode_count >= self.max_episodes:
            if self.verbose > 0:
                logger.info(
                    "Stopping training: reached %d episodes.", self.episode_count
                )
            return False
        return True
