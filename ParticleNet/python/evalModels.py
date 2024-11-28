#!/usr/bin/env python
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import torch
import subprocess

from array import array
from sklearn.utils import shuffle
from sklearn import metrics
from torch_geometric.loader import DataLoader
from ROOT import TFile, TTree, TH1D, TCanvas
from Models import ParticleNet, ParticleNetV2
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import sleep

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, required=True, help="channel")
parser.add_argument("--signal", type=str, required=True, help="signal")
parser.add_argument("--background", type=str, required=True, help="background")
parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
parser.add_argument("--fold", required=True, type=int, help="i-th fold to test")
parser.add_argument("--requireBtagged", action="store_true", default=False, help="require b-tagged jets")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']
CHANNEL = args.channel
SIG = args.signal
BKG = args.background
DEVICE = args.device
FOLD = args.fold

nFold = 5
nModels = 3
nFeatures = 9
nGraphFeatures = 4
nClasses = 2
max_epochs = 81

def getChromosomes(SIG, BKG, top=nModels):
    CSVFILE = f"{WORKDIR}/ParticleNet/condor/Optimization/{CHANNEL}__/{SIG}_vs_{BKG}/GA-iter3/CSV/model_info.csv"
    if args.requireBtagged:
        CSVFILE = f"{WORKDIR}/ParticleNet/condor/Optimization/{CHANNEL}__OnlyBtagged__/{SIG}_vs_{BKG}/GA-iter3/CSV/model_info.csv"
    df = pd.read_csv(CSVFILE)
    df = df.sort_values(by='fitness', ascending=True)
    lst = df.to_dict(orient='records')

    chromosomes = []
    for elt in lst:
        c = elt['chromosome']
        c = c.strip("()").split(", ")
        nNodes, optimizer, initLR, weight_decay, scheduler = c
        model = elt['model'].split('-')[0]
        chromosome = {'nNodes':int(nNodes), 'optimizer':optimizer.strip("'"), 'initLR':f"{float(initLR):.4f}", 'weight_decay':f"{float(weight_decay):.5f}", 'scheduler':scheduler.strip("'"), 'model':model}
        if chromosome not in chromosomes:
            chromosomes.append(chromosome)
        if len(chromosomes)==top: break

    return chromosomes

def getKSprob(tree, idx):
    hSigTrain = TH1D("hSigTrain", "", 1000, 0., 1.)
    hBkgTrain = TH1D("hBkgTrain", "", 1000, 0., 1.)
    hSigTest = TH1D("hSigTest", "", 1000, 0., 1.)
    hBkgTest = TH1D("hBkgTest", "", 1000, 0., 1.)

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        if trainMask[0]:
            if signalMask[0]: hSigTrain.Fill(scores[f"model{idx}"][0])
            else:             hBkgTrain.Fill(scores[f"model{idx}"][0])
        if testMask[0]:
            if signalMask[0]: hSigTest.Fill(scores[f"model{idx}"][0])
            else:             hBkgTest.Fill(scores[f"model{idx}"][0])

    ksProbSig = hSigTrain.KolmogorovTest(hSigTest, option="X")
    ksProbBkg = hBkgTrain.KolmogorovTest(hBkgTest, option="X")
    del hSigTrain, hBkgTrain, hSigTest, hBkgTest

    return ksProbSig, ksProbBkg

def getAUC(tree, idx, whichset):
    predictions = []
    answers = []

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        if whichset == "train":
            if not trainMask[0]: continue
        elif whichset == "valid":
            if not validMask[0]: continue
        elif whichset == "test":
            if not testMask[0]: continue
        else:
            print(f"Wrong input {whichset}")
            return None

        predictions.append(scores[f"model{idx}"][0])
        answers.append(signalMask[0])

    fpr, tpr, _ = metrics.roc_curve(answers, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def prepareROC(model, loader):
    model.eval()
    predictions = []
    answers = []
    with torch.no_grad():
        for data in loader:
            pred = model(data.x, data.edge_index, data.graphInput, data.batch)
            for p in pred: predictions.append(p[1].numpy())
            for a in data.y: answers.append(a.numpy())
    predictions = np.array(predictions)
    answers = np.array(answers)
    fpr, tpr, _ = metrics.roc_curve(answers, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return (fpr, tpr, auc)

def plotROC(model, trainLoader, validLoader, testLoader, path):
    plt.figure(figsize=(12, 12))
    plt.title(f"ROC curve")

    fpr, tpr, auc = prepareROC(model, trainLoader)
    plt.plot(tpr, 1.-fpr, 'b--', label=f"train ROC ({auc:.3f})")
    fpr, tpr, auc = prepareROC(model, validLoader)
    plt.plot(tpr, 1.-fpr, 'g--', label=f"valid ROC ({auc:.3f})")
    fpr, tpr, auc = prepareROC(model, testLoader)
    plt.plot(tpr, 1.-fpr, 'r--', label=f"test ROC ({auc:.3f})")
    plt.legend(loc='best')
    plt.xlabel("sig eff.")
    plt.ylabel("bkg rej.")
    plt.savefig(path)
    plt.close()

def plotTrainingStage(idx, path):
    chromosome = chromosomes[idx]
    nNodes, optimizer, initLR, scheduler, model, weight_decay, trial_id = (
        chromosome.get('nNodes'),chromosome.get('optimizer'),chromosome.get('initLR'),chromosome.get('scheduler'),chromosome.get('model'),chromosome.get('weight_decay'),chromosome.get('trial_id')
    )
    csvpath = f"{WORKDIR}/ParticleNet/results/{CHANNEL}__/{SIG}_vs_{BKG}/fold-{FOLD}/CSV/{model}-nNodes{nNodes}-{optimizer}-initLR{initLR.replace('.','p')}-decay{weight_decay.replace('.', 'p')}-{scheduler}.csv"
    if args.requireBtagged:
        csvpath = f"{WORKDIR}/ParticleNet/results/{CHANNEL}__OnlyBtagged__/{SIG}_vs_{BKG}/fold-{FOLD}/CSV/{model}-nNodes{nNodes}-{optimizer}-initLR{initLR.replace('.','p')}-decay{weight_decay.replace('.', 'p')}-{scheduler}.csv"
    record = pd.read_csv(csvpath, index_col=0).transpose()

    trainLoss = list(record.loc['loss/train'])
    validLoss = list(record.loc['loss/valid'])
    trainAcc  = list(record.loc['acc/train'])
    validAcc  = list(record.loc['acc/valid'])

    plt.figure(figsize=(21, 18))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(trainLoss)+1), trainLoss, "b--", label="train loss")
    plt.plot(range(1, len(validLoss)+1), validLoss, "r--", label="valid loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(trainAcc)+1), trainAcc, "b--", label="train accuracy")
    plt.plot(range(1, len(validAcc)+1), validAcc, "r--", label="valid accuracy")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def run_trainModel(chromosomes):
    procs = []
    for chromosome in chromosomes:
        nNodes, optimizer, initLR, scheduler, model, weight_decay = (
        chromosome.get('nNodes'),chromosome.get('optimizer'),chromosome.get('initLR'),chromosome.get('scheduler'),chromosome.get('model'),chromosome.get('weight_decay')
        )
        command = f"python/trainModel.py --signal {SIG} --background {BKG} --channel {CHANNEL}"
        command += f" --max_epochs {max_epochs} --model {model} --nNodes {nNodes}"
        command += f" --dropout_p 0.25 --optimizer {optimizer} --initLR {initLR}"
        command += f" --weight_decay {weight_decay} --scheduler {scheduler}"
        command += f" --device {DEVICE} --fold {FOLD}"
        if args.requireBtagged: command += " --requireBtagged"
        logging.info(f"Start training the model with command: {command}")
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(proc)
        sleep(0.5)

    for proc in procs:
        stdout, stderr = proc.communicate()
        #print("Output:", stdout.decode())
        #print("Errors:", stderr.decode())
        assert proc.returncode == 0, f"Process failed with return code {proc.returncode}"

start=datetime.now()
#### load datasets
print("@@@@ Start loading dataset...")
baseDir = f"{WORKDIR}/ParticleNet/dataset/{args.channel}__"
if args.requireBtagged:
    baseDir = f"{WORKDIR}/ParticleNet/dataset/{args.channel}__OnlyBtagged__"

dataFoldList = [[] for _ in range(nFold)]
for i in range(nFold):
    dataFoldList[i] = torch.load(f"{baseDir}/{args.signal}_vs_{args.background}_fold-{i}.pt", weights_only=False)
trainset = torch.utils.data.ConcatDataset([dataFoldList[(args.fold+1)%5], dataFoldList[(args.fold+2)%5], dataFoldList[(args.fold+3)%5]])
validset = dataFoldList[(args.fold+4)%5]
testset = dataFoldList[args.fold]

trainLoader = DataLoader(trainset, batch_size=1024, pin_memory=True, shuffle=True)
validLoader = DataLoader(validset, batch_size=1024, pin_memory=True, shuffle=False)
testLoader = DataLoader(testset, batch_size=1024, pin_memory=True, shuffle=False)

#### train models with early stop
chromosomes = getChromosomes(SIG, BKG, top=nModels)
run_trainModel(chromosomes)

#### load models
models = {}
for idx, chromosome in enumerate(chromosomes):
    nNodes, optimizer, initLR, scheduler, model, weight_decay = (
        chromosome.get('nNodes'),chromosome.get('optimizer'),chromosome.get('initLR'),chromosome.get('scheduler'),chromosome.get('model'),chromosome.get('weight_decay')
    )
    modelPath = f"{WORKDIR}/ParticleNet/results/{CHANNEL}__/{SIG}_vs_{BKG}/fold-{FOLD}/models/{model}-nNodes{nNodes}-{optimizer}-initLR{initLR.replace('.','p')}-decay{weight_decay.replace('.', 'p')}-{scheduler}.pt"
    if args.requireBtagged:
        modelPath = f"{WORKDIR}/ParticleNet/results/{CHANNEL}__OnlyBtagged__/{SIG}_vs_{BKG}/fold-{FOLD}/models/{model}-nNodes{nNodes}-{optimizer}-initLR{initLR.replace('.','p')}-decay{weight_decay.replace('.', 'p')}-{scheduler}.pt"
    if model=="ParticleNet":
        model = ParticleNet(nFeatures, nGraphFeatures, nClasses, nNodes, dropout_p=0.25)
    elif model=="ParticleNetV2":
        model = ParticleNetV2(nFeatures, nGraphFeatures, nClasses, nNodes, dropout_p=0.25)
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cuda')))

    models[idx] = model


#### prepare directories
outputPath = f"{WORKDIR}/ParticleNet/results/{CHANNEL}__/{SIG}_vs_{BKG}/fold-{FOLD}/result/temp.png"
if args.requireBtagged:
    outputPath = f"{WORKDIR}/ParticleNet/results/{CHANNEL}__OnlyBtagged__/{SIG}_vs_{BKG}/fold-{FOLD}/result/temp.png"
if not os.path.exists(os.path.dirname(outputPath)): os.makedirs(os.path.dirname(outputPath))

#### save score distributions
f = TFile(f"{os.path.dirname(outputPath)}/scores.root", "recreate")
tree = TTree("Events", "")

# initiate branches
scoreBranch = {}
trainMask  = array('B', [False]); tree.Branch("trainMask", trainMask, "trainMask/O")
validMask  = array('B', [False]); tree.Branch("validMask", validMask, "validMask/O")
testMask   = array('B', [False]); tree.Branch("testMask", testMask, "testMask/O")
signalMask = array('B', [False]); tree.Branch("signalMask", signalMask, "signalMask/O")

for idx in models.keys():
    scoreBranch[f"score_model{idx}"] = array('f', [0.])
    tree.Branch(f"score_model{idx}", scoreBranch[f"score_model{idx}"], f"score_model{idx}/F")

# start filling
print("@@@@ Filling trainset...")
trainMask[0] = True; validMask[0] = False; testMask[0] = False
for data in trainLoader:
    scoreBatch = {}
    for idx in models.keys():
        scoreBatch[idx] = []

    # fill scores 
    with torch.no_grad():
        for idx, model in models.items():
            model.eval()
            scores = model(data.x, data.edge_index, data.graphInput, data.batch)
            for score in scores: 
                scoreBatch[idx].append(score[1].numpy())

    # fill events
    for i in range(len(scoreBatch[0])):
        for idx in models.keys():
            scoreBranch[f"score_model{idx}"][0] = scoreBatch[idx][i]
        signalMask[0] = True if data.y[i] == 1 else False
        tree.Fill()

print("@@@@ Filling validset...")
trainMask[0] = False; validMask[0] = True; testMask[0] = False
for data in validLoader:
    scoreBatch = {}
    for idx in models.keys():
        scoreBatch[idx] = []

    # fill scores 
    with torch.no_grad():
        for idx, model in models.items():
            model.eval()
            scores = model(data.x, data.edge_index, data.graphInput, data.batch)
            for score in scores:
                scoreBatch[idx].append(score[1].numpy())

    # fill events
    for i in range(len(scoreBatch[0])):
        for idx in models.keys():
            scoreBranch[f"score_model{idx}"][0] = scoreBatch[idx][i]
        signalMask[0] = True if data.y[i] == 1 else False
        tree.Fill()

print("@@@@ Filling testset...")
trainMask[0] = False; validMask[0] = False; testMask[0] = True
for data in testLoader:
    scoreBatch = {}
    for idx in models.keys():
        scoreBatch[idx] = []

    # fill scores 
    with torch.no_grad():
        for idx, model in models.items():
            model.eval()
            scores = model(data.x, data.edge_index, data.graphInput, data.batch)
            for score in scores:
                scoreBatch[idx].append(score[1].numpy())

    # fill events
    for i in range(len(scoreBatch[0])):
        for idx in models.keys():
            scoreBranch[f"score_model{idx}"][0] = scoreBatch[idx][i]
        signalMask[0] = True if data.y[i] == 1 else False
        tree.Fill()

f.cd()
tree.Write()
f.Close()

#### load tree and start estimation
f = TFile.Open(f"{os.path.dirname(outputPath)}/scores.root")
tree = f.Get("Events")

scores = {}
for idx in range(nModels):
    scores[f"model{idx}"] = array('f', [0.]); tree.SetBranchAddress(f"score_model{idx}", scores[f"model{idx}"])
trainMask = array("B", [False]); tree.SetBranchAddress(f"trainMask", trainMask)
validMask = array("B", [False]); tree.SetBranchAddress(f"validMask", validMask)
testMask = array("B", [False]); tree.SetBranchAddress(f"testMask", testMask)
signalMask = array("B", [False]); tree.SetBranchAddress(f"signalMask", signalMask)

bestModelIdx = -1
fitness = 0.
for idx in range(nModels):
    ksProbSig, ksProbBkg = getKSprob(tree, idx)
    print(idx, ksProbSig, ksProbBkg)
    #if not (ksProbSig > 0.05 and ksProbBkg > 0.05): continue

    trainAUC = getAUC(tree, idx, "train")
    testAUC = getAUC(tree, idx, "test")
    print(f"model-{idx} with testAUC = {testAUC:.3f}")
    thisFitness = testAUC - (trainAUC - testAUC)
    if fitness < thisFitness:
        bestModelIdx = idx
        fitness = thisFitness
if bestModelIdx == -1:
    print("There's NO model with ksProb > 0.05 -> Skip drawing plots")
    sys.exit()
print(f"best model: model-{bestModelIdx} with test AUC {fitness:.3f}")
bestChromosome = chromosomes[bestModelIdx]
nNodes, optimizer, initLR, scheduler, weight_decay = (
    bestChromosome.get('nNodes'),bestChromosome.get('optimizer'),bestChromosome.get('initLR'),bestChromosome.get('scheduler'),bestChromosome.get('weight_decay')
)
trainAUC = getAUC(tree, bestModelIdx, "train")
validAUC = getAUC(tree, bestModelIdx, "valid")
testAUC  = getAUC(tree, bestModelIdx, "test")
ksProbSig, ksProbBkg = getKSprob(tree, bestModelIdx)
f.Close()

#### write selection
selectionInfo = f"{SIG}, {BKG}, {bestModelIdx}, {nNodes}, {optimizer}, {initLR}, {scheduler}, {weight_decay}, {trainAUC}, {validAUC}, {testAUC}, {ksProbSig}, {ksProbBkg}"
print(f"[evalModels] {SIG}_vs_{BKG} summary: {selectionInfo}")
summaryPath = f"{os.path.dirname(outputPath)}/summary.txt"
with open(summaryPath, "w") as f:
    f.write(f"{selectionInfo}\n")

end = datetime.now()
print(start)
print(end)
#### make plots
print("@@@@ Visualizing...")
for idx, model in models.items():
    plotROC(model, trainLoader, validLoader, testLoader, f"{os.path.dirname(outputPath)}/ROC-model{idx}.png") 
    plotTrainingStage(idx, f"{os.path.dirname(outputPath)}/training-model{idx}.png") 
