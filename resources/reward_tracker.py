class RewardTracker:
    def __init__(self):
        self.moving_sum = 0
        self.moving_squared_sum = 0
        self.count = 0

    def add(self, reward):
        self.moving_sum += reward
        self.moving_squared_sum += reward**2
        self.count += 1

    def mean(self):
        return self.moving_sum / self.count

    def std_dev(self):
        return (self.moving_squared_sum / self.count - self.mean()**2)**0.5