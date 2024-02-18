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

    def push(self, dqn_input, dqn_output, td_error):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (dqn_input, dqn_output)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        dqn_input, dqn_output = zip(*batch)  # Unpack pairs of (x, y)
        return np.array(dqn_input), np.array(dqn_output)

    def __len__(self):
        return len(self.buffer)


