import numpy as np


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_value = np.inf
        self.not_improving_epochs = 0
        self.early_stop = False

    def update(self, value):
        if value + self.delta >= self.best_value:
            if self.not_improving_epochs == self.patience:
                self.early_stop = True
            else:
                self.not_improving_epochs += 1
        else:
            self.not_improving_epochs = 0
            self.best_value = value

    def __call__(self, value):
        self.update(value)
