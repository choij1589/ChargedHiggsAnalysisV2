#!/usr/bin/env python
import os
import argparse
import json
import ROOT
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True, help="2016preVFP, 2016postVFP, 2017, 2018, FullRun2")
parser.add_argument("--channel", type=str, required=True, help="SR1E2Mu, SR3Mu, Combined")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
args = parser.parse_args()

if args.method == "Baseline":
    MASSPOINTs = ["MHc-70_MA-15", "MHc-70_MA-40", "MHc-70_MA-65",
                  "MHc-100_MA-15", "MHc-100_MA-60", "MHc-100_MA-95",
                  "MHc-130_MA-15", "MHc-130_MA-55", "MHc-130_MA-90", "MHc-130_MA-125",
                  "MHc-160_MA-15", "MHc-160_MA-85", "MHc-160_MA-120", "MHc-160_MA-155"]
elif args.method == "ParticleNet":
    MASSPOINTs = ["MHc-100_MA-95", "MHc-130_MA-90", "MHc-160_MA-85"]
else:
    raise ValueError("Invalid method")
REFERENCE_XSEC = 5.


def parseAsymptoticLimit(masspoint, method):
    base_dir = f"templates/{args.era}/{args.channel}/{masspoint}/Shape/{method}"
    f = ROOT.TFile.Open(f"{base_dir}/higgsCombineTest.AsymptoticLimits.mH120.root")
    limit = f.Get("limit")
    xsecs = {}
    for idx, entry in enumerate(limit):
        xsecs[idx] = entry.limit*REFERENCE_XSEC
    f.Close()

    out = {}
    out["exp-2"] = xsecs[0]
    out["exp-1"] = xsecs[1]
    out["exp0"] = xsecs[2]
    out["exp+1"] = xsecs[3]
    out["exp+2"] = xsecs[4]
    out["obs"] = xsecs[5]
    
    return out

def readHybridNewResult(path):
    f = ROOT.TFile.Open(path)
    limit = f.Get("limit")
    try:
        for entry in limit:
            out = entry.limit
    except Exception as e:
        print(e)
    f.Close()

    return out*REFERENCE_XSEC

def parseHybridNewLimit(masspoint, method):
    base_dir = f"templates/{args.era}/{args.channel}/{masspoint}/Shape/{method}"
    out = {}
    out["exp-2"] = readHybridNewResult(f"{base_dir}/higgsCombineTest.HybridNew.mH120.quant0.025.root")
    out["exp-1"] = readHybridNewResult(f"{base_dir}/higgsCombineTest.HybridNew.mH120.quant0.160.root")
    out["exp0"] = readHybridNewResult(f"{base_dir}/higgsCombineTest.HybridNew.mH120.quant0.500.root")
    out["exp+1"] = readHybridNewResult(f"{base_dir}/higgsCombineTest.HybridNew.mH120.quant0.840.root")
    out["exp+2"] = readHybridNewResult(f"{base_dir}/higgsCombineTest.HybridNew.mH120.quant0.975.root")
    out["obs"] = readHybridNewResult(f"{base_dir}/higgsCombineTest.HybridNew.mH120.root")

    return out

if __name__ == "__main__":
    limits = {}
    for masspoint in MASSPOINTs:
        mA = masspoint.split("_")[1].split("-")[1]
        if mA == 15:
            if masspoint != "MHc-70_MA-15":
                continue
        limits[mA] = parseAsymptoticLimit(masspoint, args.method)

    with open(f"results/json/limits.{args.era}.{args.channel}.Asymptotic.{args.method}.json", "w") as f:
        json.dump(limits, f, indent=2)

