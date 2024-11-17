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
else:
    raise f"Wrong channel {args.channel}"
# Good to make up / down histograms in this place?
theorySysts = [
        ("AlpS_up", "AlpS_down"),
        ("AlpSfact_up", "AlpSfact_down"),
        tuple([f"PDFReweight_{i}" for i in range(100)]),
        tuple([f"ScaleVar_{i}" for i in [0, 1, 2, 3, 4, 6, 8]]),
        tuple([f"PSVar_{i}" for i in range(4)])
        ]


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
central = getHist(args.masspoint, mA, width, sigma); f.cd(); central.Write()
for syst in promptSysts:
    hist = getHist(args.masspoint, mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
    hist = getHist(args.masspoint, mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

# Make up / down histograms for AlpS variations
#hist = getHist(args.masspoint, mA, width, sigma, "AlpS_up"); hist.SetName("AlpSUp"); f.cd(); hist.Write()
#hist = getHist(args.masspoint, mA, width, sigma, "AlpS_down"); hist.SetName("AlpSDown"); f.cd(); hist.Write()

# Make up / down histograms for PDF variations
# first make histograms for each PDF variation, then make up / down histograms
hists_pdf = []
for i in range(100):
    hists_pdf.append(getHist(args.masspoint, mA, width, sigma, f"PDFReweight_{i}"))
# calculate RMS of PDF variations
pdf_up = central.Clone(f"{args.masspoint}_PDFUp")
pdf_down = central.Clone(f"{args.masspoint}_PDFDown")
for i in range(1, central.GetNbinsX()+1):
    bin_values = np.array([hist.GetBinContent(i) for hist in hists_pdf])
    rms = np.std(bin_values, ddof=1)
    pdf_up.SetBinContent(i, central.GetBinContent(i) + rms)
    pdf_down.SetBinContent(i, central.GetBinContent(i) - rms)
f.cd(); pdf_up.Write(); pdf_down.Write()

# Make up / down histograms for scale variations
hists_scale = []
for i in [0, 1, 2, 3, 4, 6, 8]:
    hists_scale.append(getHist(args.masspoint, mA, width, sigma, f"ScaleVar_{i}"))
# calculate min/max of scale variations
scale_up = central.Clone(f"{args.masspoint}_ScaleUp")
scale_down = central.Clone(f"{args.masspoint}_ScaleDown")
for i in range(1, central.GetNbinsX()+1):
    bin_values = np.array([hist.GetBinContent(i) for hist in hists_scale])
    scale_up.SetBinContent(i, np.max(bin_values))
    scale_down.SetBinContent(i, np.min(bin_values))
f.cd(); scale_up.Write(); scale_down.Write()

# Make up / down histograms for PS variations
hists_ps = []
for i in range(4):
    hists_ps.append(getHist(args.masspoint, mA, width, sigma, f"PSVar_{i}"))
# calculate min/max of PS variations
ps_up = central.Clone(f"{args.masspoint}_PSUp")
ps_down = central.Clone(f"{args.masspoint}_PSDown")
for i in range(1, central.GetNbinsX()+1):
    bin_values = np.array([hist.GetBinContent(i) for hist in hists_ps])
    ps_up.SetBinContent(i, np.max(bin_values))
    ps_down.SetBinContent(i, np.min(bin_values))
f.cd(); ps_up.Write(); ps_down.Write()

logging.info("Processing nonprompt")
hist = getHist("nonprompt", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()

logging.info("Processing conversion")
hist = getHist("conversion", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()

logging.info("Processing diboson")
for process in ["WZ", "ZZ"]:
    hist = getHist(process, mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
    for syst in promptSysts:
        hist = getHist(process, mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
        hist = getHist(process, mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

#hist = getHist("diboson", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
#for syst in promptSysts:
#    hist = getHist("diboson", mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
#    hist = getHist("diboson", mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

logging.info("Processing ttX")
for process in ["ttW", "ttZ", "ttH"]:
    hist = getHist(process, mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
    for syst in promptSysts:
        hist = getHist(process, mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
        hist = getHist(process, mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

#hist = getHist("ttX", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
#for syst in promptSysts:
#    hist = getHist("ttX", mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
#    hist = getHist("ttX", mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

logging.info("Processing others")
hist = getHist("others", mA, width, sigma); data_obs.Add(hist); f.cd(); hist.Write()
for syst in promptSysts:
    hist = getHist("others", mA, width, sigma, f"{syst}Up"); f.cd(); hist.Write()
    hist = getHist("others", mA, width, sigma, f"{syst}Down"); f.cd(); hist.Write()

f.cd(); data_obs.Write()
f.Close()
