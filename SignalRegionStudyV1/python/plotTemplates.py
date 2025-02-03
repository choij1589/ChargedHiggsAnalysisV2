#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from math import pow, sqrt
from plotter import ComparisonCanvas

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="2016preVFP / 2016postVFP / 2017 / 2018")
parser.add_argument("--channel", required=True, type=str, help="SR1E2Mu / SR3Mu")
parser.add_argument("--masspoint", required=True, type=str, help="signal mass point")
parser.add_argument("--method", required=True, type=str, help="Baseline / ParticleNet")
args = parser.parse_args()

WORKDIR = os.environ["WORKDIR"]

config = {
    "era": args.era,
    "xTitle": "m_{#mu^{+}#mu^{-}} [GeV]",
    "yTitle": "Events",
    "yRange": [0., 2.]
}

# nonprompt, conversion, others are already converged
diboson = ["WZ", "ZZ"]
ttX = ["ttW", "ttZ", "ttH", "tZq"]
promptBkgs = diboson + ttX + ["others"]

# Systematics
if args.channel == "SR1E2Mu":
    SYSTs = [("L1PrefireUp", "L1PrefireDown"),
             ("PileupReweightUp", "PileupReweightDown"),
             ("MuonIDSFUp", "MuonIDSFDown"),
             ("ElectronIDSFUp", "ElectronIDSFDown"),
             ("TriggerSFUp", "TriggerSFDown"),
             ("JetEnUp", "JetEnDown"),
             ("JetResUp", "JetResDown"),
             ("ElectronEnUp", "ElectronEnDown"),
             ("ElectronResUp", "ElectronResDown"),
             ("MuonEnUp", "MuonEnDown")]
elif args.channel == "SR3Mu":
    SYSTs = [("L1PrefireUp", "L1PrefireDown"),
             ("PileupReweightUp", "PileupReweightDown"),
             ("MuonIDSFUp", "MuonIDSFDown"),
             ("TriggerSFUp", "TriggerSFDown"),
             ("JetEnUp", "JetEnDown"),
             ("JetResUp", "JetResDown"),
             ("MuonEnUp", "MuonEnDown")]
else:
    raise ValueError("Invalid channel")

# Load histograms
HISTs = {}
COLORs = {}

def addHist(name, hist, histDict):
    if hist is None:
        return
    if histDict[name] is None:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)
            
rtfile_path = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}/shapes_input.noera.root"
assert os.path.exists(rtfile_path), f"File not found: {rtfile_path}"
rtfile = ROOT.TFile.Open(rtfile_path, "read")

data_obs = rtfile.Get("data_obs"); data_obs.SetDirectory(0)

## signal
SIGs = {}
signal = rtfile.Get(args.masspoint)
SIGs[args.masspoint] = signal.Clone(args.masspoint)
SIGs[args.masspoint].SetDirectory(0)

## backgrounds
nonprompt = rtfile.Get("nonprompt")
for bin in range(nonprompt.GetNcells()):
    nonprompt.SetBinError(bin, nonprompt.GetBinContent(bin)*0.3)
HISTs["nonprompt"] = nonprompt.Clone("nonprompt")
HISTs["nonprompt"].SetDirectory(0)

conversion = rtfile.Get("conversion")
for bin in range(conversion.GetNcells()):
    conversion.SetBinError(bin, conversion.GetBinContent(bin)*0.2)
HISTs["conversion"] = conversion.Clone("conversion")
HISTs["conversion"].SetDirectory(0)

for sample in promptBkgs:
    try:
        h = rtfile.Get(sample); h.SetDirectory(0)
    except:
        print(f"{sample} not fould for {args.era}/{args.channel}/{args.masspoint}")
        HISTs[sample] = None
        continue
    hSysts = []
    for syst_up, syst_down in SYSTs:
        try:
            h_up = rtfile.Get(f"{sample}_{syst_up}"); h_up.SetDirectory(0)
            h_down = rtfile.Get(f"{sample}_{syst_down}"); h_down.SetDirectory(0)
            hSysts.append((h_up, h_down))
        except:
            print(f"Systematic {syst_up} or {syst_down} not found for {sample}")
            continue
    
    for bin in range(h.GetNcells()):
        stat_unc = h.GetBinError(bin)
        envelops = []
        for h_up, h_down in hSysts:
            systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
            systDown = abs(h_down.GetBinContent(bin) - h.GetBinContent(bin))
            envelops.append(max(systUp, systDown))
        total_unc = sqrt(pow(stat_unc, 2) + sum([pow(x, 2) for x in envelops]))
        h.SetBinError(bin, total_unc)
    HISTs[sample] = h.Clone(sample)
    HISTs[sample].SetDirectory(0)
rtfile.Close()
    
temp_dict = {}
temp_dict["nonprompt"] = None
temp_dict["conversion"] = None
temp_dict["ttX"] = None
temp_dict["diboson"] = None
temp_dict["others"] = None

addHist("nonprompt", HISTs["nonprompt"], temp_dict)
addHist("conversion", HISTs["conversion"], temp_dict)
for sample in ttX:
    addHist("ttX", HISTs[sample], temp_dict)
for sample in diboson:
    addHist("diboson", HISTs[sample], temp_dict)
addHist("others", HISTs["others"], temp_dict)

# filter out none historgrams from temp_dict
BKGs = {name: hist for name, hist in temp_dict.items() if hist}

COLORs["data"] = ROOT.kBlack
COLORs[args.masspoint] = ROOT.kRed
COLORs["nonprompt"] = ROOT.kGray+2
COLORs["ttX"] = ROOT.kBlue
COLORs["conversion"] = ROOT.kViolet
COLORs["diboson"] = ROOT.kGreen
COLORs["others"] = ROOT.kOrange
c = ComparisonCanvas(config=config)
c.drawSignals(SIGs, COLORs)
c.drawBackgrounds(BKGs, COLORs)
c.drawData(data_obs)
c.drawRatio()
c.drawLegend()
c.finalize()

hist_path = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}/template.png"
c.SaveAs(hist_path)
