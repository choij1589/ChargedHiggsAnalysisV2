#!/usr/bin/env python

"""
Search for the best hyperparameters for the training with a configuration given by GA
Use 0, 2, 4 folds for training, 1, 3 folds for validation
"""

import os
import argparse
import logging

import ROOT
from sklearn.utils import shuffle
from itertools import product

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd
import matplotlib as plt
from array import array
from sklearn import metrics

from Preprocess import GraphDataset
from Preprocess import rtfileToDataList
from Models import ParticleNet, ParticleNetV2
from MLTools import EarlyStopper, SummaryWriter

#### parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--signal", required=True, type=str, help="signal")
parser.add_argument("--background", required=True, type=str, help="background")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--iter", required=True, type=int, help="GA iteration number")
parser.add_argument("--idx", required=True, type=int, help="GA index number")
parser.add_argument("--max_epochs", default=81, type=int, help="max epochs")
parser.add_argument("--model", default="ParticleNet", type=str, help="model type")
parser.add_argument("--nNodes", required=True, type=int, help="number of nodes for each layer")
parser.add_argument("--dropout_p", default=0.25, type=float, help="dropout_p")
parser.add_argument("--optimizer", required=True, type=str, help="optimizer")
parser.add_argument("--initLR", required=True, type=float, help="initial learning rate")
parser.add_argument("--weight_decay", required=True, type=float, help="weight decay")
parser.add_argument("--scheduler", required=True, type=str, help="lr scheduler")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument("--pilot", action="store_true", default=False, help="pilot mode")
parser.add_argument("--requireBtagged", action="store_true", default=False, help="read dataset from b-tagged samples")
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

# check arguments
if args.channel not in ["Skim1E2Mu", "Skim3Mu", "Combined"]:
    raise ValueError(f"Invalid channel {args.channel}")
if args.signal not in ["MHc-100_MA-95", "MHc-130_MA-90", "MHc-160_MA-85"]:
    raise ValueError(f"Invalid signal {args.signal}")
if args.background not in ["nonprompt", "diboson", "ttZ"]:
    raise ValueError(f"Invalid background {args.background}")

WORKDIR = os.environ["WORKDIR"]

def transform_data(data):
    theta = torch.rand(data.x.size(0), device=args.device) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([c, -s, s, c], dim=1).view(-1, 2, 2)

    data.x[:, 1:3] = torch.bmm(R, data.x[:, 1:3].unsqueeze(-1)).squeeze(-1)

    parity_mask = torch.rand(data.x.size(0), device=args.device) > 0.5
    data.x[parity_mask, 1:4] *= -1

#### load dataset
logging.info("Start loading dataset")
baseDir = f"{WORKDIR}/ParticleNet/dataset/{args.channel}__"
if args.requireBtagged:
    baseDir += "OnlyBtagged__"
if args.pilot:
    baseDir += "pilot__"

nFold = 5
dataFoldList = [[] for _ in range(nFold)]
for i in range(nFold):
    dataFoldList[i] = torch.load(f"{baseDir}/{args.signal}_vs_{args.background}_fold-{i}.pt", weights_only=False)
trainset = torch.utils.data.ConcatDataset([dataFoldList[0], dataFoldList[2], dataFoldList[4]])
validset = torch.utils.data.ConcatDataset([dataFoldList[1], dataFoldList[3]])

trainLoader = DataLoader(trainset, batch_size=1024, pin_memory=True, shuffle=True)
validLoader = DataLoader(validset, batch_size=1024, pin_memory=True, shuffle=False)

if "cuda" in args.device:
    logging.info("Using cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

#### helper functions
def train(model, optimizer, scheduler, use_plateau_scheduler=False):
    model.train()

    total_loss = 0.
    for data in trainLoader:
        transform_data(data.to(args.device))
        out = model(data.x, data.edge_index, data.graphInput, data.batch)
        optimizer.zero_grad()
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if use_plateau_scheduler:
        avg_loss = total_loss / len(trainLoader.dataset)
        scheduler.step(total_loss)
    else:
        scheduler.step()

def test(model, loader):
    model.eval()

    loss = 0.
    correct = 0.
    with torch.no_grad():
        for data in loader:
            out = model(data.x.to(args.device), data.edge_index.to(args.device), data.graphInput.to(args.device), data.batch.to(args.device))
            pred = out.argmax(dim=1)
            answer = data.y.to(args.device)
            loss += float(F.cross_entropy(out, answer).sum())
            correct += int((pred == answer).sum())
    loss /= len(loader.dataset)
    correct /= len(loader.dataset)

    return (loss, correct)

def main():
    ## setup
    logging.info(f"Using model {args.model}")
    nFeatures = 9
    nGraphFeatures = 4
    nClasses = 2
    if args.model == "ParticleNet":
        model = ParticleNet(nFeatures, nGraphFeatures, nClasses, args.nNodes, args.dropout_p).to(args.device)
    elif args.model == "ParticleNetV2":
        model = ParticleNetV2(nFeatures, nGraphFeatures, nClasses, args.nNodes, args.dropout_p).to(args.device)
    else:
        raise NotImplementedError(f"Unsupporting model {args.model}")

    logging.info(f"Using optimizer {args.optimizer}")
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.initLR, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initLR, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.initLR, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Unsupporting optimizer {args.optimizer}")
    #optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    logging.info(f"Using scheduler {args.scheduler}")
    if args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    elif args.scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    elif args.scheduler == "CyclicLR":
        cycle_momentum = True if args.optimizer == "RMSprop" else False
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.initLR/5., max_lr=args.initLR*2,
                                                      step_size_up=3, step_size_down=5, cycle_momentum=cycle_momentum)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        raise NotImplementedError(f"Unsupporting scheduler {args.scheduler}")

    modelName = f"{args.model}-model{args.idx}"
    #modelName =  f"{args.model}-nNodes{args.nNodes}-{args.optimizer}-initLR{str(format(args.initLR, '.4f')).replace('.','p')}-decay{str(format(args.weight_decay, '.5f')).replace('.', 'p')}-{args.scheduler}"
    logging.info("Start training...")
    checkptpath = f"{WORKDIR}/ParticleNet/results/{args.channel}__/{args.signal}_vs_{args.background}/GA-iter{args.iter}/models/{modelName}.pt"
    summarypath = f"{WORKDIR}/ParticleNet/results/{args.channel}__/{args.signal}_vs_{args.background}/GA-iter{args.iter}/CSV/{modelName}.csv"
    if args.requireBtagged:
        checkptpath = f"{WORKDIR}/ParticleNet/results/{args.channel}__OnlyBtagged__/{args.signal}_vs_{args.background}/GA-iter{args.iter}/models/{modelName}.pt"
        summarypath = f"{WORKDIR}/ParticleNet/results/{args.channel}__OnlyBtagged__/{args.signal}_vs_{args.background}/GA-iter{args.iter}/CSV/{modelName}.csv"
    earlyStopper = EarlyStopper(patience=7, path=checkptpath)
    summaryWriter = SummaryWriter(name=modelName)

    for epoch in range(args.max_epochs):
        train(model, optimizer, scheduler, use_plateau_scheduler=(args.scheduler=="ReduceLROnPlateau"))
        trainLoss, trainAcc = test(model, trainLoader)
        validLoss, validAcc = test(model, validLoader)
        summaryWriter.addScalar("loss/train", trainLoss)
        summaryWriter.addScalar("loss/valid", validLoss)
        summaryWriter.addScalar("acc/train", trainAcc)
        summaryWriter.addScalar("acc/valid", validAcc)

        logging.info(f"[EPOCH {epoch}]\tTrain Acc: {trainAcc*100:.2f}%\tTrain Loss: {trainLoss:.4e}")
        logging.info(f"[EPOCH {epoch}]\tValid Acc: {validAcc*100:.2f}%\tValid Loss: {validLoss:.4e}")

        #penalty = max(0, validLoss-trainLoss)
        earlyStopper.update(validLoss, model)
        if earlyStopper.earlyStop:
            logging.info(f"Early stopping in epoch {epoch}"); break
        print()

    summaryWriter.to_csv(summarypath)


if __name__ == "__main__":
    main()
