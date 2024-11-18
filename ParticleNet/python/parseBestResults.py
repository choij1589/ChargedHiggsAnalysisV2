#!/usr/bin/env python
## parseBestResults.py
## In this script, we parse the best results of each fold to ParticleNet/results directory
import os, shutil
import glob
import argparse

parser = argparse.ArgumentParser(description='Parse results to SKFlat')
parser.add_argument("--channel", type=str, required=True, help="Skim1E2Mu, Skim3Mu, Combined")
parser.add_argument("--signal", type=str, required=True, help="signal mass point")
parser.add_argument("--background", type=str, required=True, help="background")
parser.add_argument("--nfold", type=int, default=5, help="Number of folds")
args = parser.parse_args()

WORKDIR = os.getenv("WORKDIR")
RESULTDIR = f"{WORKDIR}/ParticleNet/condor/Evaluation/{args.channel}/{args.signal}_vs_{args.background}"


# Read summary.txt
def retrieve_model_name(fold):
    with open(f"{RESULTDIR}/fold-{fold}/summary.txt", "r") as f:
        summary = f.readline().strip().split(", ")
    signal, background, model_idx, nnode, optimizer, init_lr, scheduler = tuple(summary[:7])

    # search model, no decay info in the summary.txt
    model_name = f"ParticleNet-nNodes{nnode}-{optimizer}-initLR{str(init_lr).replace('.', 'p')}-decay*-{scheduler}.pt"
    model_path = f"{RESULTDIR}/fold-{fold}/models/{model_name}"
    files = glob.glob(model_path)
    if len(files) > 1:
        print("Warning: More than one model found")
    return files[0]

if __name__ == "__main__":
    for fold in range(args.nfold):
        result_path = f"{WORKDIR}/ParticleNet/results/{args.channel}/{args.signal}_vs_{args.background}/fold-{fold}"
        os.makedirs(result_path, exist_ok=True)

        with open(f"{RESULTDIR}/fold-{fold}/summary.txt", "r") as f:
            summary = f.readline().strip().split(", ")
        signal, background, model_idx, nnode, optimizer, init_lr, scheduler = tuple(summary[:7])
  
        model_path = retrieve_model_name(fold)
        training_info = model_path.replace("models", "CSV").replace(".pt", ".csv")
        rtfile_path = model_path.replace("models", "trees").replace(".pt", ".root")

        shutil.copy(f"{RESULTDIR}/fold-{fold}/summary.txt", f"{result_path}/summary.txt")
        shutil.copy(model_path, f"{result_path}/ParticleNet.pt")
        shutil.copy(training_info, f"{result_path}/training_info.csv")
        shutil.copy(rtfile_path, f"{result_path}/score.root")
        shutil.copy(f"{RESULTDIR}/fold-{fold}/result/ROC-model{model_idx}.png", f"{result_path}/ROC.png")
        shutil.copy(f"{RESULTDIR}/fold-{fold}/result/training-model{model_idx}.png", f"{result_path}/training.png")



