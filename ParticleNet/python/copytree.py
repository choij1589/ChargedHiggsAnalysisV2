#!/usr/bin/env python
import os, shutil
import logging
import argparse
from itertools import product, combinations
from ROOT import TFile
logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--channel", required=True, type=str, help="channel name")
args = parser.parse_args()
# global variables
WORKDIR = os.environ['WORKDIR']
CHANNEL = args.channel

# no. of events to copy
SIGNALs = ["MHc-160_MA-85", "MHc-130_MA-90", "MHc-100_MA-95"]
NONPROMPTs = ["TTLL_powheg"]
DIBOSONs = ["WZTo3LNu_mllmin0p1_powheg", "ZZTo4L_powheg"]
TTZ = ["ttZToLLNuNu"]
BACKGROUNDs = NONPROMPTs + DIBOSONs + TTZ

ERAs = ["2016preVFP", "2016postVFP", "2017", "2018"]
nEvtsToCopy = {"signal": [15000, 15000, 30000, 45000],
               "TTLL_powheg": [15000, 15000, 30000, 45000],
               "WZTo3LNu_mllmin0p1_powheg": [7500, 7500, 15000, 22500],
               "ZZTo4L_powheg": [7500, 7500, 15000, 22500],
               "ttZToLLNuNu": [15000, 15000, 30000, 45000]}

#### make directories
for era in ERAs:
    os.makedirs(f"dataset/{era}/{CHANNEL}", exist_ok=True)

#### clean output directories
os.system(f"rm -rf dataset/*/{CHANNEL}/*")

##### signal
for era, signal in product(ERAs, SIGNALs):
    nEvts = nEvtsToCopy["signal"][ERAs.index(era)]
    # check no. of events
    f = TFile.Open(f"/DATA/SKFlat/DataPreprocess/{era}/{CHANNEL}__/DataPreprocess_TTToHcToWAToMuMu_{signal}.root")
    tree = f.Get("Events")
    try:
        assert nEvts < tree.GetEntries()
    except:
        logging.warning(f"Small no. of events for {era}-{signal}: nEvts({nEvts}) < nEntries({tree.GetEntries()})")
        logging.warning(f"Force nEvts to no. of tree entries")
        nEvts = tree.GetEntries()
    f.Close()

    # copyfile
    os.system(f"{WORKDIR}/ParticleNet/libs/copytree /DATA/SKFlat/DataPreprocess/{era}/{CHANNEL}__/DataPreprocess_TTToHcToWAToMuMu_{signal}.root {nEvts}")
    shutil.move(f"/DATA/SKFlat/DataPreprocess/{era}/{CHANNEL}__/DataPreprocess_TTToHcToWAToMuMu_{signal}_copy.root", f"dataset/{era}/{CHANNEL}/DataPreprocess_TTToHcToWAToMuMu_{signal}.root")

##### background
for era, bkg in product(ERAs, BACKGROUNDs):
    nEvts = nEvtsToCopy[bkg][ERAs.index(era)]
    # check no. of events
    f = TFile.Open(f"/DATA/SKFlat/DataPreprocess/{era}/{CHANNEL}__/DataPreprocess_{bkg}.root")
    tree = f.Get("Events")
    try:
        assert nEvts < tree.GetEntries()
    except:
        logging.warning(f"Small no. of events for {era}-{bkg}: nEvts({nEvts}) < nEntries({tree.GetEntries()})")
        logging.warning(f"Force nEvts to no. of tree entries")
        nEvts = tree.GetEntries()
    f.Close()

    # copy trees
    os.system(f"{WORKDIR}/ParticleNet/libs/copytree /DATA/SKFlat/DataPreprocess/{era}/{CHANNEL}__/DataPreprocess_{bkg}.root {nEvts}")
    shutil.move(f"/DATA/SKFlat/DataPreprocess/{era}/{CHANNEL}__/DataPreprocess_{bkg}_copy.root", f"dataset/{era}/{CHANNEL}/DataPreprocess_{bkg}.root")

## hadd backgrounds
for era in ERAs:
    # hadd nonprompt
    shutil.move(f"dataset/{era}/{CHANNEL}/DataPreprocess_TTLL_powheg.root", f"dataset/{era}/{CHANNEL}/DataPreprocess_nonprompt.root")
    #os.system(f"hadd -f dataset/{era}/{CHANNEL}/DataPreprocess_nonprompt.root dataset/{era}/{CHANNEL}/DataPreprocess_DYJetsToMuMu_MiNNLO.root dataset/{era}/{CHANNEL}/DataPreprocess_TTLL_powheg.root")

    # hadd diboson
    os.system(f"hadd -f dataset/{era}/{CHANNEL}/DataPreprocess_diboson.root dataset/{era}/{CHANNEL}/DataPreprocess_WZTo3LNu_mllmin0p1_powheg.root dataset/{era}/{CHANNEL}/DataPreprocess_ZZTo4L_powheg.root")

    # rename ttZ
    shutil.move(f"dataset/{era}/{CHANNEL}/DataPreprocess_ttZToLLNuNu.root", f"dataset/{era}/{CHANNEL}/DataPreprocess_ttZ.root")
