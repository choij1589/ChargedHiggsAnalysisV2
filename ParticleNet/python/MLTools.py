import os
import numpy as np
import pandas as pd
import torch

class EarlyStopper():
    def __init__(self, patience=7, delta=0, path="./checkpoint.pt"):
        self.patience = patience
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valLossMin = np.Inf
        self.delta = delta
        self.path = path
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    def update(self, valLoss, panelty,  model):
        score = -(valLoss+panelty)
        if self.bestScore is None:
            self.bestScore = score
            self.saveCheckpoint(valLoss, model)
        elif score <= self.bestScore + self.delta:
            self.counter += 1
            print(f"[EarlyStopping counter] {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.bestScore = score
            self.saveCheckpoint(valLoss, model)
            self.counter = 0

    def saveCheckpoint(self, valLoss, model):
        torch.save(model.state_dict(), self.path)
        self.valLossMin = valLoss

class SummaryWriter():
    def __init__(self, name):
        self.name = name
        self.scalarDict = {}

    def addScalar(self, key, value):
        if not key in self.scalarDict.keys(): self.scalarDict[key] = []
        self.scalarDict[key].append(value)

    def getScalar(self, key):
        return self.scalarDict[key]

    def to_csv(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        df = pd.DataFrame(self.scalarDict)
        df.to_csv(path)
