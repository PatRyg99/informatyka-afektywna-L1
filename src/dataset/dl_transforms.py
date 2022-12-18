class InterpolateGraphs:
    def __init__(self, rate: float = 0.2):
        self.rate = rate

    def __call__(self, batch):

        for i in range(len(batch)):
            x1, x2 = batch[i], batch[(i + 1) % len(batch)]

            delta = x2.pos - x1.pos
            x1.pos += self.rate * delta

        return batch
