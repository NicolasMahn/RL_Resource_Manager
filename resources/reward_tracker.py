class RewardTracker:
    def __init__(self):
        # Initialize RewardTracker with zeroed statistics
        self.moving_sum = 0  # Sum of all rewards
        self.moving_squared_sum = 0  # Sum of squares of all rewards
        self.count = 0  # Total number of rewards added

    def add(self, reward):
        # Add a new reward to the tracker
        self.moving_sum += reward  # Update the sum of rewards
        self.moving_squared_sum += reward**2  # Update the sum of squared rewards
        self.count += 1  # Increment the count of rewards

    def mean(self):
        # Calculate the mean of the rewards
        return self.moving_sum / self.count if self.count != 0 else 0

    def std_dev(self):
        # Calculate the standard deviation of the rewards
        mean = self.mean()  # Calculate the mean of rewards
        return ((self.moving_squared_sum / self.count - mean**2)**0.5) if self.count != 0 else 0
