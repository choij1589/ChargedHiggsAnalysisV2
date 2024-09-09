#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from math import pow, sqrt
from plotter import ComparisonCanvas
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="EMu / DiMu")
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--no_lepton_correction", action="store_true", default=False, help="no_lepton_correction")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

with open("histConfigs.json") as f:
    config = json.load(f)[args.histkey]
config["era"] = args.era

#### Sample list
DATASTREAM = ""
if args.channel == "DiMu":
    DATASTREAM = "DoubleMuon"
elif args.channel == "EMu":
    DATASTREAM = "MuonEG"
else:
    raise ValueError(f"invalid channel {args.channel}")

W  = ["WJets_MG"]
DY = ["DYJets", "DYJets10to50_MG"]
TT = ["TTLL_powheg", "TTLJ_powheg"]
VV = ["WW_pythia", "WZ_pythia", "ZZ_pythia"]
ST = ["SingleTop_sch_Lep", "SingleTop_tch_top_Incl", "SingleTop_tch_antitop_Incl",
      "SingleTop_tW_top_NoFullyHad", "SingleTop_tW_antitop_NoFullyHad"]
MCList = W + DY + TT + VV + ST

#### Systematics
if args.channel == "DiMu":
    SYSTs = [("L1PrefireUp", "L1PrefireDown"),
             ("PileupReweightUp", "PileupReweightDown"),
             ("MuonIDSFUp", "MuonIDSFDown"),
             ("DblMuTrigSFUp", "DblMuTrigSFDown"),
             ("MuonEnUp", "MuonEnDown"),
             ("ElectronEnUp", "ElectronEnDown"),
             ("ElectronResUp", "ElectronResDown"),
             ("JetEnUp", "JetEnDown"),
             ("JetResUp", "JetResDown")]
elif args.channel == "EMu":
    SYSTs = [("L1PrefireUp", "L1PrefireDown"),
             ("PileupReweightUp", "PileupReweightDown"),
             ("MuonIDSFUp", "MuonIDSFDown"),
             ("ElectronIDSFUp", "ElectronIDSFDown"),
             ("EMuTrigSFUp", "EMuTrigSFDown"),
             ("MuonEnUp", "MuonEnDown"),
             ("ElectronEnUp", "ElectronEnDown"),
             ("ElectronResUp", "ElectronResDown"),
             ("JetEnUp", "JetEnDown"),
             ("JetResUp", "JetResDown")]
    
#### Get Histograms
HISTs = {}
COLORs = {}

## DATA
## hadd data if there is no combined file
file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/CR_DiLepton/{args.era}/Run{args.channel}__/DATA/CR_DiLepton_DoubleMuon.root"
logging.debug(f"file_path: {file_path}")
if not os.path.exists(file_path):
    logging.info(os.listdir(f"{os.path.dirname(file_path)}"))
    logging.info(f"file {file_path} does not exist. hadding...")
    response = input("Do you want to continue? [y/n]: ").strip().lower()
    if response == "y":
        os.system(f"hadd -f {file_path} {os.path.dirname(file_path)}/CR_DiLepton_{DATASTREAM}_*.root")
    elif response == "n":
        print("No data file to proceed plotting, exiting...")
        exit(1)
    else:
        raise ValueError("invalid response")

assert os.path.exists(file_path)
f = ROOT.TFile.Open(file_path)
data = f.Get(f"{args.channel}/Central/{args.histkey}"); data.SetDirectory(0)
f.Close()

for sample in MCList:
    file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/CR_DiLepton/{args.era}/Run{args.channel}__RunSyst__/CR_DiLepton_{sample}.root"
    logging.debug(f"file_path: {file_path}")
    assert os.path.exists(file_path)
    f = ROOT.TFile.Open(file_path)
    if args.no_lepton_correction:
        try:
            h = f.Get(f"{args.channel}/Central_BaseWeight/{args.histkey}"); h.SetDirectory(0)
            f.Close()
        except Exception as e:
            logging.debug(e, sample)
            continue
    else:
        # Get Central histogram first. If it does not exist, skip the sample
        try:
            h = f.Get(f"{args.channel}/Central/{args.histkey}"); h.SetDirectory(0)
        except Exception as e:
            logging.debug(e, sample)
            continue
        # Get systematic histograms
        # Do not skip the sample if the systematic histogram does not exists. Skip the systematic histogram and eras from SYSTs list
        hSysts = []
        for systUp, systDown in SYSTs:
            try:
                h_up = f.Get(f"{args.channel}/{systUp}/{args.histkey}"); h_up.SetDirectory(0)
                h_down = f.Get(f"{args.channel}/{systDown}/{args.histkey}"); h_down.SetDirectory(0)
            except Exception as e:
                logging.debug(e, sample)
                continue
            hSysts.append((h_up, h_down))
        f.Close()
        
        # estimate total unc. bin by bin
        for bin in range(h.GetNcells()):
            stat_unc = h.GetBinError(bin)
            envelops = []
            for h_up, h_down in hSysts:
                systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
                systDown = abs(h_down.GetBinContent(bin) - h.GetBinContent(bin))
                envelops.append(max(systUp, systDown))
            total_uc = sqrt(pow(stat_unc, 2) + pow(max(envelops), 2))
            h.SetBinError(bin, total_unc)
    HISTs[sample] = h.Clone(sample)

#### merge background
def add_hist(name, hist, histDict):
    # content of dictionary should be initialized as "None"
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)
        
temp_dict = {}
temp_dict["W"]  = None
temp_dict["DY"] = None
temp_dict["TT"] = None
temp_dict["VV"] = None
temp_dict["ST"] = None

for sample in W:
    if not sample in HISTs.keys(): continue
    add_hist("W", HISTs[sample], temp_dict)
for sample in DY:
    if not sample in HISTs.keys(): continue
    add_hist("DY", HISTs[sample], temp_dict)
for sample in TT:
    if not sample in HISTs.keys(): continue
    add_hist("TT", HISTs[sample], temp_dict)
for sample in VV:
    if not sample in HISTs.keys(): continue
    add_hist("VV", HISTs[sample], temp_dict)
for sample in ST:
    if not sample in HISTs.keys(): continue
    add_hist("ST", HISTs[sample], temp_dict)
    
#### remove none histograms
BKGs = {}
for key, value in temp_dict.items():
    if value is None: continue
    BKGs[key] = value
    
COLORs["data"] = ROOT.kBlack
COLORs["W"]  = ROOT.kMagenta
COLORs["DY"] = ROOT.kGray
COLORs["TT"] = ROOT.kViolet
COLORs["VV"] = ROOT.kGreen
COLORs["ST"] = ROOT.kAzure

#### draw plots
c = ComparisonCanvas(config=config)
c.drawBackgrounds(BKGs, COLORs)
c.drawData(data)
c.drawRatio()
c.drawLegend()
c.finalize()

output_path = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/AfterLeptonCorrection/{args.histkey.replace('/', '_')}.png"
if args.no_lepton_correction:
    output_path = output_path.replace("After", "Before")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
c.SaveAs(output_path)
