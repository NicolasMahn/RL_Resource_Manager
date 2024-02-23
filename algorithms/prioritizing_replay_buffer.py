import random
import numpy as np

# Random number generators
rnd = np.random.random
randint = np.random.randint


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity

    def _get_priority(self, td_error):
        return (np.abs(td_error) + 1e-5) ** self.alpha

    def add(self, state, action, reward, next_state, td_error):
        priority = self._get_priority(td_error)
        self.tree.add(priority, (state, action, reward, next_state))

    def sample(self, batch_size, beta=0.4):
        states = []
        actions = []
        rewards = []
        next_states = []
        idxs = []
        segment = self.tree.total_priority() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total_priority()
        is_weight = np.power(self.tree.total_priority() * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(idxs)

    def update(self, idx, td_error):
        priority = self._get_priority(td_error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.size
