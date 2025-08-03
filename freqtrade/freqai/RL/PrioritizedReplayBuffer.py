import numpy as np


class PrioritizedReplayBuffer:
    """Basic prioritized experience replay buffer.

    Sampling probability of each experience is proportional to its priority.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-3,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=float)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples, strict=False)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities, strict=False):
            self.priorities[idx] = prio + self.epsilon

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)
