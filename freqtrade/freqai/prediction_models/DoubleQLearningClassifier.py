import random
from collections import defaultdict
from typing import Any

import numpy as np

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)


class DoubleQLearningClassifier(BaseReinforcementLearningModel):
    """Double Q-learning classifier."""

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
        self,
        state: tuple[float, ...],
        q1: dict[tuple[float, ...], np.ndarray],
        q2: dict[tuple[float, ...], np.ndarray],
    ) -> int:
        if random.random() < self.epsilon:
            return self.train_env.action_space.sample()
        values = q1[state] + q2[state]
        return int(np.argmax(values))

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        env = self.train_env
        n_actions = env.action_space.n
        q1: dict[tuple[float, ...], np.ndarray] = defaultdict(lambda: np.zeros(n_actions))
        q2: dict[tuple[float, ...], np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

        for _ in range(self.train_cycles):
            state, _ = env.reset()
            done = False
            while not done:
                s_key = tuple(state)
                action = self._epsilon_action(s_key, q1, q2)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ns_key = tuple(next_state)
                if random.random() < 0.5:
                    best_next = int(np.argmax(q1[ns_key]))
                    q1[s_key][action] += self.alpha * (
                        reward + self.gamma * q2[ns_key][best_next] - q1[s_key][action]
                    )
                else:
                    best_next = int(np.argmax(q2[ns_key]))
                    q2[s_key][action] += self.alpha * (
                        reward + self.gamma * q1[ns_key][best_next] - q2[s_key][action]
                    )
                state = next_state
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return _DoubleQPolicy(q1, q2)


class _DoubleQPolicy:
    """Policy wrapper for double Q-learning tables."""

    def __init__(
        self, q1: dict[tuple[float, ...], np.ndarray], q2: dict[tuple[float, ...], np.ndarray]
    ) -> None:
        self.q1 = q1
        self.q2 = q2

    def predict(self, obs: Any, deterministic: bool = True):
        state = tuple(obs)
        q1_val = self.q1.get(state)
        q2_val = self.q2.get(state)
        if q1_val is None:
            q1_val = np.zeros(len(next(iter(self.q1.values()))))
        if q2_val is None:
            q2_val = np.zeros(len(next(iter(self.q1.values()))))
        q_values = q1_val + q2_val
        action = int(np.argmax(q_values))
        return np.array([action]), None
