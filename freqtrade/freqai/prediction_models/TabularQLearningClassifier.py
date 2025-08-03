import random
from collections import defaultdict
from typing import Any

import numpy as np

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)


class TabularQLearningClassifier(BaseReinforcementLearningModel):
    """Simple tabular Q-learning classifier."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = self.freqai_info.get("rl_config", {})
        self.alpha: float = cfg.get("alpha", 0.1)
        self.gamma: float = cfg.get("gamma", 0.99)
        self.epsilon: float = cfg.get("epsilon", 1.0)
        self.epsilon_decay: float = cfg.get("epsilon_decay", 0.995)
        self.min_epsilon: float = cfg.get("min_epsilon", 0.01)
        self.train_cycles: int = cfg.get("train_cycles", 1)

    def _epsilon_action(
        self, state: tuple[float, ...], q_table: dict[tuple[float, ...], np.ndarray]
    ) -> int:
        if random.random() < self.epsilon:
            return self.train_env.action_space.sample()
        return int(np.argmax(q_table[state]))

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        env = self.train_env
        n_actions = env.action_space.n
        q_table: dict[tuple[float, ...], np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

        for _ in range(self.train_cycles):
            state, _ = env.reset()
            done = False
            while not done:
                s_key = tuple(state)
                action = self._epsilon_action(s_key, q_table)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ns_key = tuple(next_state)
                q_current = q_table[s_key][action]
                q_next = np.max(q_table[ns_key])
                q_table[s_key][action] = q_current + self.alpha * (
                    reward + self.gamma * q_next - q_current
                )
                state = next_state
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return _QTablePolicy(q_table)


class _QTablePolicy:
    """Policy wrapper for tabular Q-values."""

    def __init__(self, table: dict[tuple[float, ...], np.ndarray]):
        self.table = table

    def predict(self, obs: Any, deterministic: bool = True):
        state = tuple(obs)
        q_values = self.table.get(state, np.zeros(len(next(iter(self.table.values())))))
        action = int(np.argmax(q_values))
        return np.array([action]), None
