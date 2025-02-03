#!/usr/bin/env python
import os
import shutil
import argparse
import re
import ROOT

parser = argparse.ArgumentParser(description='Update era suffix in the input file')
parser.add_argument("--era", type=str, required=True, help="era")
parser.add_argument("--channel", type=str, required=True, help="channel")
parser.add_argument("--masspoint", type=str, required=True, help="masspoint")
parser.add_argument("--method", type=str, required=True, help="Baseline / ParticleNet")
args = parser.parse_args()

TEMPLATE_DIR = f"templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}"

# BackUp the ROOT file
shutil.copy(f"{TEMPLATE_DIR}/shapes_input.root", f"{TEMPLATE_DIR}/shapes_input.noera.root")

# Read the datacard
with open(f"{TEMPLATE_DIR}/datacard.txt", "r") as f:
    datacard_lines = f.readlines()

# Extract systematics with era suffix
if args.era == "2016preVFP":
    era_suffix = "_16a"
elif args.era == "2016postVFP":
    era_suffix = "_16b"
elif args.era == "2017":
    era_suffix = "_17"
elif args.era == "2018":
    era_suffix = "_18"
else:
    raise ValueError(f"Invalid era: {args.era}")

# Extract systematics with era suffix
systematics_for_update = []
for line in datacard_lines:
    if era_suffix in line:
        systematics_for_update.append(line.split()[0].split("_")[0])

# Update the ROOT file
original = ROOT.TFile.Open(f"{TEMPLATE_DIR}/shapes_input.noera.root")
new = ROOT.TFile.Open(f"{TEMPLATE_DIR}/shapes_input.root", "RECREATE")

for key in original.GetListOfKeys():
    systematic_token = key.GetName().split("_") # e.g. others_JetResDown
    #if len(systematic_token) < 2: continue      # e.g. others
    this_syst = systematic_token[-1].replace("Up", "").replace("Down", "")
    if this_syst in systematics_for_update:
        if "Up" in key.GetName():
            new_key = key.GetName().replace("Up", f"{era_suffix}Up")
        elif "Down" in key.GetName():
            new_key = key.GetName().replace("Down", f"{era_suffix}Down")
        else:
            raise ValueError(f"Invalid systematic type to update with era suffix: {key.GetName()}")
        h = original.Get(key.GetName())
        h_clone = h.Clone(new_key)
        new.cd()
        h_clone.Write()
    else:
        h = original.Get(key.GetName())
        h_clone = h.Clone()
        new.cd()
        h_clone.Write()
original.Close()
new.Close()
