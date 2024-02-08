import random
import numpy as np

# Random number generators
rnd = np.random.random
randint = np.random.randint


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, X, y):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (X, y)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        X, y = zip(*batch)  # Unpack pairs of (X, y)
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.buffer)


