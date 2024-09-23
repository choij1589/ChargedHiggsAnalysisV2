#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import numpy as np
import pandas as pd
import joblib
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--masspoint", required=True, type=str, help="masspoint")
parser.add_argument("--optimize", action="store_true", default=False, help="do ParticleNet optimization")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.getenv("WORKDIR")
BACKGROUNDs = ["nonprompt", "conversion", "diboson", "ttX", "others"]
if args.channel == "SR1E2Mu":
    promptSysts = ["L1Prefire", "PileupReweight",
                   "MuonIDSF", "ElectronIDSF", "TriggerSF",
                   "JetRes", "JetEn", "MuonEn", "ElectronRes", "ElectronEn"]
elif args.channel == "SR3Mu":
    promptSysts = ["L1Prefire", "PileupReweight",
                   "MuonIDSF", "TriggerSF",
                   "JetRes", "JetEn", "MuonEn"]


def getFitResult(input_path, output_path, mA):
    fitter = ROOT.AmassFitter(input_path, output_path)
    fitter.fitMass(mA, mA-20., mA+20.)
    fitter.saveCanvas(f"{base_path}/fit_result.png")
    mA = fitter.getRooMA().getVal()
    width = fitter.getRooWidth().getVal()
    sigma = fitter.getRooSigma().getVal()
    fitter.Close()
    return mA, width, sigma

def getHist(process, mA, width, sigma, syst="Central"):
    logging.debug(process, syst)
    if syst == "Central":
        hist = ROOT.TH1D(process, "", 15, mA-5*width-3*sigma, mA+5*width+3*sigma)
    else:
        hist = ROOT.TH1D(f"{process}_{syst}", "", 15, mA-5*width-3*sigma, mA+5*width+3*sigma)
    f = ROOT.TFile.Open(f"{WORKDIR}/SignalRegionStudy/samples/{args.era}/{args.channel.replace('SR', 'Skim')}/{args.masspoint}/{process}.root")
    tree = f.Get(f"{process}_{syst}")

    for evt in tree:
        if args.channel == "SR1E2Mu":
            hist.Fill(evt.mass1, evt.weight)
        elif args.channel == "SR3Mu":
            hist.Fill(evt.mass1, evt.weight)
            hist.Fill(evt.mass2, evt.weight)
    hist.SetDirectory(0)

    return hist

if args.optimize:
    base_path = f"{WORKDIR}/SignalRegionStudy/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/ParticleNet"
else:
    base_path = f"{WORKDIR}/SignalRegionStudy/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/Baseline"
os.makedirs(base_path, exist_ok=True)

f = ROOT.TFile(f"{base_path}/shapes_input.root", "RECREATE")
mA = float(args.masspoint.split("_")[1].split("-")[1])
# fit
input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel.replace('SR', 'Skim')}__/PromptSkimmer_TTToHcToWAToMuMu_{args.masspoint}.root"
output_path = f"{base_path}/fit_result.root"
mA, width, sigma = getFitResult(input_path, output_path, mA)

data_obs = ROOT.TH1D("data_obs", "", 15, mA-5*width-3*sigma, mA+5*width+3*sigma)

logging.info(f"Processing {args.masspoint}")
hist = getHist(args.masspoint, mA, width, sigma); f.cd(); hist.Write()
for syst in promptSysts:
    hist = getHist(args.masspoint, mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
    hist = getHist(args.masspoint, mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

logging.info("Processing nonprompt")
hist = getHist("nonprompt", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()

logging.info("Processing conversion")
hist = getHist("conversion", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()

logging.info("Processing diboson")
hist = getHist("diboson", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
for syst in promptSysts:
    hist = getHist("diboson", mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
    hist = getHist("diboson", mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

logging.info("Processing ttX")
hist = getHist("ttX", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
for syst in promptSysts:
    hist = getHist("ttX", mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
    hist = getHist("ttX", mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

logging.info("Processing others")
hist = getHist("others", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
for syst in promptSysts:
    hist = getHist("others", mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
    hist = getHist("others", mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

f.cd(); data_obs.Write()
f.Close()
