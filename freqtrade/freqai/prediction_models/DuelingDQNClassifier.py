from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.PrioritizedReplayBuffer import PrioritizedReplayBuffer


class DuelingQNetwork(nn.Module):
    """Simple dueling network with separate value and advantage streams."""

    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        val = self.value_stream(x)
        adv = self.adv_stream(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DuelingDQNClassifier(BaseClassifierModel):
    """Deep Q-network with dueling architecture and prioritized replay."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = self.freqai_info.get("rl_config", {})
        self.gamma: float = cfg.get("gamma", 0.99)
        self.epsilon: float = cfg.get("epsilon", 1.0)
        self.epsilon_decay: float = cfg.get("epsilon_decay", 0.995)
        self.min_epsilon: float = cfg.get("min_epsilon", 0.01)
        self.batch_size: int = cfg.get("batch_size", 32)
        self.train_cycles: int = cfg.get("train_cycles", 1)
        self.buffer_size: int = cfg.get("buffer_size", 10_000)
        self.lr: float = cfg.get("learning_rate", 1e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions: int = 0

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        X = data_dictionary["train_features"].to_numpy()
        y = data_dictionary["train_labels"].to_numpy()[:, 0].astype(int)
        self.n_actions = len(np.unique(y))
        input_dim = X.shape[1]
        policy_net = DuelingQNetwork(input_dim, self.n_actions).to(self.device)
        target_net = DuelingQNetwork(input_dim, self.n_actions).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.Adam(policy_net.parameters(), lr=self.lr)
        buffer = PrioritizedReplayBuffer(self.buffer_size)

        for _ in range(self.train_cycles):
            for i in range(len(X) - 1):
                state = X[i]
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    with torch.no_grad():
                        q_vals = policy_net(
                            torch.tensor(state, dtype=torch.float32, device=self.device)
                        )
                        action = int(torch.argmax(q_vals).item())
                reward = 1.0 if action == y[i] else 0.0
                next_state = X[i + 1]
                done = i == len(X) - 2
                buffer.add(state, action, reward, next_state, done)

                if len(buffer) >= self.batch_size:
                    (
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        idxs,
                        weights,
                    ) = buffer.sample(self.batch_size)
                    states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
                    actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
                    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                    next_states_t = torch.tensor(
                        next_states, dtype=torch.float32, device=self.device
                    )
                    dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
                    weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

                    q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = torch.argmax(policy_net(next_states_t), dim=1)
                        next_q = (
                            target_net(next_states_t)
                            .gather(1, next_actions.unsqueeze(1))
                            .squeeze(1)
                        )
                        target = rewards_t + self.gamma * next_q * (1 - dones_t)
                    td_error = target - q_values
                    loss = (weights_t * td_error.pow(2)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    buffer.update_priorities(idxs, td_error.detach().cpu().numpy())

            target_net.load_state_dict(policy_net.state_dict())
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return _DuelingPolicy(policy_net, self.device, self.n_actions)


class _DuelingPolicy:
    """Wrapper providing predict and predict_proba for the trained network."""

    def __init__(self, network: DuelingQNetwork, device: torch.device, n_actions: int) -> None:
        self.network = network
        self.device = device
        self.classes_ = np.arange(n_actions)

    def predict(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            q_vals = self.network(
                torch.tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            )
            actions = torch.argmax(q_vals, dim=1).cpu().numpy()
        return actions

    def predict_proba(self, obs: Any) -> np.ndarray:
        with torch.no_grad():
            q_vals = (
                self.network(torch.tensor(np.asarray(obs), dtype=torch.float32, device=self.device))
                .cpu()
                .numpy()
            )
        exp_q = np.exp(q_vals - np.max(q_vals, axis=1, keepdims=True))
        return exp_q / exp_q.sum(axis=1, keepdims=True)
