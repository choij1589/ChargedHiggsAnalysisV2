import os
import logging
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)

class EarlyStopper():
    def __init__(self, patience=7, improvement=0.005, path="./checkpoint.pt"):
        self.patience = patience
        self.improvement = improvement
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valLossMin = np.inf
        self.delta = 0.
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def update(self, valLoss, model):
        score = -valLoss
        if self.bestScore is None:
            self.bestScore = score
            self.delta = self.bestScore * self.improvement
            self.saveCheckpoint(valLoss, model)
        elif score <= self.bestScore + self.delta:
            self.counter += 1
            logging.info(f"[EarlyStopping counter] {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.earlyStop = True
        else:
            self.bestScore = score
            self.delta = self.bestScore*self.improvement
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
