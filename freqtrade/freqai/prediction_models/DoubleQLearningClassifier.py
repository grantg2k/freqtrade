import random
from collections import defaultdict
from typing import Any

import numpy as np

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


class DoubleQLearningClassifier(BaseClassifierModel):
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
        self.n_actions: int = 0

    def _epsilon_action(
        self,
        state: tuple[float, ...],
        q1: dict[tuple[float, ...], np.ndarray],
        q2: dict[tuple[float, ...], np.ndarray],
    ) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        values = q1[state] + q2[state]
        return int(np.argmax(values))

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0].astype(int)
        self.n_actions = len(np.unique(y))
        q1: dict[tuple[float, ...], np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))
        q2: dict[tuple[float, ...], np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))

        for _ in range(self.train_cycles):
            for i in range(len(X) - 1):
                s_key = tuple(X[i])
                ns_key = tuple(X[i + 1])
                action = self._epsilon_action(s_key, q1, q2)
                reward = 1.0 if action == y[i] else 0.0
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
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return _DoubleQPolicy(q1, q2, self.n_actions)


class _DoubleQPolicy:
    """Policy wrapper for double Q-learning tables."""

    def __init__(
        self,
        q1: dict[tuple[float, ...], np.ndarray],
        q2: dict[tuple[float, ...], np.ndarray],
        n_actions: int,
    ) -> None:
        self.q1 = q1
        self.q2 = q2
        self.n_actions = n_actions
        self.classes_ = np.arange(n_actions)

    def _get_q(self, state: tuple[float, ...]) -> np.ndarray:
        q1_val = self.q1.get(state, np.zeros(self.n_actions))
        q2_val = self.q2.get(state, np.zeros(self.n_actions))
        return q1_val + q2_val

    def predict(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        arr = np.asarray(obs)
        actions = [int(np.argmax(self._get_q(tuple(row)))) for row in arr]
        return np.array(actions)

    def predict_proba(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs)
        prob_list = []
        for row in arr:
            q_vals = self._get_q(tuple(row))
            exp_q = np.exp(q_vals - np.max(q_vals))
            prob_list.append(exp_q / exp_q.sum())
        return np.vstack(prob_list)
