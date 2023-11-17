import random
import numpy as np

# Random number generators
rnd = np.random.random
randint = np.random.randint


class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize the Replay Buffer with a specified capacity
        self.capacity = capacity
        self.buffer = []  # The buffer to store experiences
        self.position = 0  # Tracks the current position to add a new experience

    def push(self, state, action, reward, next_state):
        # Add an experience to the buffer
        if len(self.buffer) < self.capacity:
            # Append a placeholder if the buffer is not full
            self.buffer.append(None)
        # Store the experience (state, action, reward, next state) in the buffer
        self.buffer[self.position] = (state, action, reward, next_state)
        # Update the position, wrapping around if at the end of the buffer
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Separate the batch into individual components
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)
