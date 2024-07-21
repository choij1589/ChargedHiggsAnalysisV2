#!/usr/bin/env python
import os
import argparse
import logging

import ROOT
from sklearn.utils import shuffle
from itertools import product

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
#from torchlars import LARS

import numpy as np
import pandas as pd
import matplotlib as plt
from array import array
from sklearn import metrics

from Preprocess import GraphDataset
from Preprocess import rtfileToDataList
from Models import ParticleNet
from MLTools import EarlyStopper, SummaryWriter

#### parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--signal", required=True, type=str, help="signal")
parser.add_argument("--background", required=True, type=str, help="background")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--epochs", required=True, type=int, help="max epochs")
parser.add_argument("--model", required=True, type=str, help="model type")
parser.add_argument("--nNodes", required=True, type=int, help="number of nodes for each layer")
parser.add_argument("--dropout_p", default=0.25, type=float, help="dropout_p")
parser.add_argument("--optimizer", required=True, type=str, help="optimizer")
parser.add_argument("--initLR", required=True, type=float, help="initial learning rate")
parser.add_argument("--scheduler", required=True, type=str, help="lr scheduler")
parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
parser.add_argument("--pilot", action="store_true", default=False, help="pilot mode")
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

#### load dataset
maxSize = 10000 if args.pilot else -1
maxSizeForEra = {
    "2016preVFP": int(maxSize / 7),
    "2016postVFP": int(maxSize / 7),
    "2017": int(maxSize * (2/7)),
    "2018": int(maxSize * (3/7))
}

logging.info("Start loading dataset")
logging.debug(maxSizeForEra)
if args.channel == "Combined":
    ## Load signal data
    sigDataList = []
    bkgDataList = []
    for era, channel in product(["2016preVFP", "2016postVFP", "2017", "2018"], ["Skim1E2Mu", "Skim3Mu"]):
        rt = ROOT.TFile.Open(f"dataset/{era}/{channel}/DataPreprocess_TTToHcToWAToMuMu_{args.signal}.root")
        sigDataList += rtfileToDataList(rt, isSignal=True, era=era, maxSize=maxSizeForEra[era])
        rt.Close()
        
        rt = ROOT.TFile.Open(f"dataset/{era}/{channel}/DataPreprocess_{args.background}.root")
        bkgDataList += rtfileToDataList(rt, isSignal=False, era=era, maxSize=maxSizeForEra[era])
        rt.Close()
    sigDataList = shuffle(sigDataList, random_state=42)
    bkgDataList = shuffle(bkgDataList, random_state=42)
else:
    sigDataList = []
    bkgDataList = []
    for era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        rt = ROOT.TFile.Open(f"dataset/{era}/{args.channel}/DataPreprocess_TTToHcToWAToMuMu_{args.signal}.root")
        sigDataList += rtfileToDataList(rt, isSignal=True, era=era, maxSize=maxSizeForEra[era])
        rt.Close()

        rt = ROOT.TFile.Open(f"dataset/{era}/{args.channel}/DataPreprocess_{args.background}.root")
        bkgDataList += rtfileToDataList(rt, isSignal=False, era=era, maxSize=maxSizeForEra[era])
        rt.Close()
    sigDataList = shuffle(sigDataList, random_state=42)
    bkgDataList = shuffle(bkgDataList, random_state=42)
logging.info("Finished loading dataset")
logging.info(f"Signal: {len(sigDataList)} events")
logging.info(f"Background: {len(bkgDataList)} events")
dataList = shuffle(sigDataList+bkgDataList, random_state=42)

trainset = GraphDataset(dataList[:int(len(dataList)*0.4)])
validset = GraphDataset(dataList[int(len(dataList)*0.4):int(len(dataList)*0.5)])
testset  = GraphDataset(dataList[int(len(dataList)*0.5):])

trainLoader = DataLoader(trainset, batch_size=1024, pin_memory=True, shuffle=True)
validLoader = DataLoader(validset, batch_size=1024, pin_memory=True, shuffle=False)
testLoader = DataLoader(testset, batch_size=1024, pin_memory=True, shuffle=False)

if "cuda" in args.device:
    logging.info("Using cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

#### helper functions
def train(model, optimizer, scheduler):
    model.train()

    for data in trainLoader:
        out = model(data.x.to(args.device), data.edge_index.to(args.device), data.graphInput.to(args.device), data.batch.to(args.device))
        loss = F.cross_entropy(out, data.y.to(args.device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
    model = ParticleNet(nFeatures, nGraphFeatures, nClasses, args.nNodes, args.dropout_p).to(args.device)

    logging.info(f"Using optimizer {args.optimizer}")
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.initLR, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initLR, weight_decay=1e-4)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.initLR, weight_decay=1e-4)
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
    else:
        raise NotImplementedError(f"Unsupporting scheduler {args.scheduler}")

    modelName =  f"{args.model}-nNodes{args.nNodes}_{args.optimizer}_initLR-{str(args.initLR).replace('.','p')}_{args.scheduler}"
    logging.info("Start training...")
    checkptpath = f"{WORKDIR}/ParticleNet/results/{args.channel}/{args.signal}_vs_{args.background}/models/{modelName}.pt"
    summarypath = f"{WORKDIR}/ParticleNet/results/{args.channel}/{args.signal}_vs_{args.background}/CSV/{modelName}.csv"
    outtreepath = f"{WORKDIR}/ParticleNet/results/{args.channel}/{args.signal}_vs_{args.background}/trees/{modelName}.root"
    earlyStopper = EarlyStopper(patience=15, path=checkptpath)
    summaryWriter = SummaryWriter(name=modelName)

    for epoch in range(args.epochs):
        train(model, optimizer, scheduler)
        trainLoss, trainAcc = test(model, trainLoader)
        validLoss, validAcc = test(model, validLoader)
        summaryWriter.addScalar("loss/train", trainLoss)
        summaryWriter.addScalar("loss/valid", validLoss)
        summaryWriter.addScalar("acc/train", trainAcc)
        summaryWriter.addScalar("acc/valid", validAcc)

        print(f"[EPOCH {epoch}]\tTrain Acc: {trainAcc*100:.2f}%\tTrain Loss: {trainLoss:.4e}")
        print(f"[EPOCH {epoch}]\tVlaid Acc: {validAcc*100:.2f}%\tValid Loss: {validLoss:.4e}\n")

        panelty = max(0, validLoss-trainLoss)
        earlyStopper.update(validLoss, panelty, model)
        if earlyStopper.earlyStop:
            print(f"Early stopping in epoch {epoch}"); break

    summaryWriter.to_csv(summarypath)

    #### save score distributions as trees
    os.makedirs(os.path.dirname(outtreepath), exist_ok=True)
    f = ROOT.TFile(outtreepath, "RECREATE")
    tree = ROOT.TTree("Events", "")
    
    # define branches
    score = array("f", [0.]); tree.Branch("score", score, "score/F")
    trainMask = array("B", [False]); tree.Branch("trainMask", trainMask, "trainMask/O")
    validMask = array("B", [False]); tree.Branch("validMask", validMask, "validMask/O")
    testMask = array("B", [False]); tree.Branch("testMask", testMask, "testMask/O")
    signalMask = array("B", [False]); tree.Branch("signalMask", signalMask, "signalMask/O")

    logging.info("Start saving score distributions")
    model.eval()
    trainMask[0] = True; validMask[0] = False; testMask[0] = False
    for i, data in enumerate(trainLoader):
        with torch.no_grad():
            out = model(data.x.to(args.device), data.edge_index.to(args.device), data.graphInput.to(args.device), data.batch.to(args.device))
            for isSignal, scoreTensor in zip(data.y, out):
                signalMask[0] = True if isSignal.numpy() else False
                score[0] = scoreTensor[1].cpu().numpy()
                tree.Fill()

    trainMask[0] = False; validMask[0] = True; testMask[0] = False
    for i, data in enumerate(validLoader):
        with torch.no_grad():
            out = model(data.x.to(args.device), data.edge_index.to(args.device), data.graphInput.to(args.device), data.batch.to(args.device))
            for signalTensor, scoreTensor in zip(data.y, out):
                signalMask[0] = True if signalTensor.numpy() else False
                score[0] = scoreTensor[1].cpu().numpy()
                tree.Fill()

    trainMask[0] = False; validMask[0] = False; testMask[0] = True
    for i, data in enumerate(testLoader):
        with torch.no_grad():
            out = model(data.x.to(args.device), data.edge_index.to(args.device), data.graphInput.to(args.device), data.batch.to(args.device))
            for signalTensor, scoreTensor in zip(data.y, out):
                signalMask[0] = True if signalTensor.numpy() else False
                score[0] = scoreTensor[1].cpu().numpy()
                tree.Fill()
    f.cd()
    tree.Write()
    f.Close()
                
if __name__ == "__main__":
    main()
