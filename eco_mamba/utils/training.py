"""Small training utilities used by the Eco-Mamba trainer."""


class AverageMeter:
    """Track a weighted running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
