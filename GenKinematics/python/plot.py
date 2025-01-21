#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from plotter import KinematicCanvas
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--mHc", type=int, default=-1, help="mHc")
args = parser.parse_args()

if args.mHc == -1:
    SIGNALs = ["MHc-70_MA-15", "MHc-100_MA-60", "MHc-130_MA-90", "MHc-160_MA-155"]
elif args.mHc == 70:
    SIGNALs = ["MHc-70_MA-15", "MHc-70_MA-40", "MHc-70_MA-65"]
elif args.mHc == 100:
    SIGNALs = ["MHc-100_MA-15", "MHc-100_MA-60", "MHc-100_MA-95"]
elif args.mHc == 130:
    SIGNALs = ["MHc-130_MA-15", "MHc-130_MA-55", "MHc-130_MA-90", "MHc-130_MA-125"]
elif args.mHc == 160:
    SIGNALs = ["MHc-160_MA-15", "MHc-160_MA-85", "MHc-160_MA-120", "MHc-160_MA-155"]
else:
    raise ValueError(f"invalid mHc {args.mHc}")

WORKDIR = os.environ['WORKDIR']
PALLATTE = [ROOT.kGray+2, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta]

with open("histConfigs.json") as f:
    config = json.load(f)[args.histkey]
config["era"] = args.era

HISTs = {}
COLORs = {}
for i, signal in enumerate(SIGNALs):
    base_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/GenKinematics/{args.era}/GenKinematics_TTToHcToWAToMuMu_{signal}.root"
    assert os.path.exists(base_path), f"file not found: {base_path}"
    f = ROOT.TFile(base_path)
    try:
        hist = f.Get(f"{args.channel}/{args.histkey}"); hist.SetDirectory(0)
        f.Close()
        HISTs[signal] = hist.Clone(signal)
        COLORs[signal] = PALLATTE[i]
    except:
        print(f"{args.histkey} not found for {signal}")
        f.Close()

canvas = KinematicCanvas(config)
canvas.drawSignals(HISTs, COLORs)
canvas.drawLegend()
canvas.finalize()

text = ROOT.TLatex()
text.SetTextSize(0.03)
text.DrawLatexNDC(0.2, 0.8, config["title"])
text.DrawLatexNDC(0.2, 0.75, "Normalized to #sigma_{sig} = 15 fb")

if args.mHc == -1:
    canvas.SaveAs(f"plots/{args.era}/{args.channel}/{args.histkey.replace('/', '_')}.png")
else:
    canvas.SaveAs(f"plots/{args.era}/{args.channel}/{args.histkey.replace('/', '_')}_mHc{args.mHc}.png")
