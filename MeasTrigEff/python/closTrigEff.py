#!/usr/bin/env python
import os
import logging
import argparse
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="Era")
parser.add_argument("--channel", required=True, type=str, help="Channel")
parser.add_argument("--process", required=True, type=str, help="Sample name")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
args = parser.parse_args()

logging.basicConfig(level = logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")

if "Run" in args.channel:
    ANALYZER = "ClosDiLepTrigs"
elif "Skim" in args.channel:
    ANALYZER = "ClosTriLepTrigs"
else:
    raise ValueError(f"Invalid channel: {args.channel}")

if "MHc" in args.process:
    SAMPLENAME = f"{ANALYZER}_TTToHcToWAToMuMu_{args.process}.root"
else:
    SAMPLENAME = f"{ANALYZER}_{args.process}.root"
file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/{ANALYZER}/{args.era}/{args.channel}__/{SAMPLENAME}"

logging.debug(f"sample name = {SAMPLENAME}")
logging.debug(f"file path = {file_path}")
assert os.path.exists(file_path), f"File not found: {file_path}"

f = ROOT.TFile.Open(file_path)
h = f.Get("sumweight"); h.SetDirectory(0); f.Close()

expected = h.GetBinContent(2) / h.GetBinContent(1)
up = h.GetBinContent(3) / h.GetBinContent(1)
down = h.GetBinContent(4) / h.GetBinContent(1)
observed = h.GetBinContent(5) / h.GetBinContent(1)

# save to json
import json
json_path = f"{WORKDIR}/MeasTrigEff/results/{args.era}/{args.channel}/{args.process}.json"
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w") as f:
    json.dump({"expected": expected, 
               "up": up, 
               "down": down, 
               "observed": observed,
               "difference": (observed-expected)/expected}, f, indent=2)

