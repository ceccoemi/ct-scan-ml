import numpy as np


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_value = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        if value + self.delta >= self.best_value:
            if self.counter == self.patience:
                self.early_stop = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_value = value
