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
parser.add_argument("--blind", default=False, action="store_true", help="blind data")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ["WORKDIR"]

with open("histConfigs.json") as f:
    config = json.load(f)[args.histkey]
config['era'] = args.era

with open(f"{WORKDIR}/CommonData/json/convScaleFactors.json") as f:
    convScaleFactors = json.load(f)[args.era]

SIGNALs = ["MHc-70_MA-15", "MHc-100_MA-60", "MHc-130_MA-90", "MHc-160_MA-155"]
if "score" in args.histkey:
    SIGNALs = [args.histkey.split("/")[0]]
    #SIGNALs = ["MHc-100_MA-95", "MHc-130_MA-90", "MHc-160_MA-85"]
CONV = ["DYJets", "DYJets10to50_MG", "ZGToLLG", "WWG", "TTG"]
DIBOSON = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
TTX = ["ttWToLNu", "ttZToLLNuNu", "ttHToNonbb", "tZq"]
OTHERs = ["WWW", "WWZ", "WZZ", "ZZZ", "tHq", "VBF_HToZZTo4L", "GluGluHToZZTo4L"]
MCList = CONV + DIBOSON + TTX + OTHERs

#### Systematics
if "1E2Mu" in args.channel:
    DATASTREAM = "MuonEG"
    SKIMFLAG = "Skim1E2Mu"
    SYSTs = [("L1PrefireUp", "L1PrefireDown"),
             ("PileupReweightUp", "PileupReweightDown"),
             ("MuonIDSFUp", "MuonIDSFDown"),
             ("ElectronIDSFUp", "ElectronIDSFDown"),
             ("TriggerSFUp", "TriggerSFDown"),
             ("JetEnUp", "JetEnDown"),
             ("JetResUp", "JetResDown"),
             ("ElectronEnUp", "ElectronEnDown"),
             ("ElectronResUp", "ElectronResDown"),
             ("MuonEnUp", "MuonEnDown"),
             ("HeavyTagUpUnCorr", "HeavyTagDownUnCorr"),
             ("HeasyTagUpCorr", "HeavyTagDownCorr"),
             ("LightTagUpUnCorr", "LightTagDownUnCorr"),
             ("LightTagUpCorr", "LightTagDownCorr")]
    ConvSF, ConvSFErr = convScaleFactors["1E2Mu"]
elif "3Mu" in args.channel:
    DATASTREAM = "DoubleMuon"
    SKIMFLAG = "Skim3Mu"
    SYSTs = [("L1PrefireUp", "L1PrefireDown"),
             ("PileupReweightUp", "PileupReweightDown"),
             ("MuonIDSFUp", "MuonIDSFDown"),
             ("TriggerSFUp", "TriggerSFDown"),
             ("JetEnUp", "JetEnDown"),
             ("JetResUp", "JetResDown"),
             ("MuonEnUp", "MuonEnDown"),
             ("HeavyTagUpUnCorr", "HeavyTagDownUnCorr"),
             ("HeasyTagUpCorr", "HeavyTagDownCorr"),
             ("LightTagUpUnCorr", "LightTagDownUnCorr"),
             ("LightTagUpCorr", "LightTagDownCorr")]
    ConvSF, ConvSFErr = convScaleFactors["3Mu"]
    if args.histkey == "nonprompt/eta": config['rebin'] = 4
    #if args.histkey == "convLep/eta": config['rebin'] = 4
else:
    raise ValueError(f"Invalide channel: {args.channel}")

logging.debug(f"DATASTREAM: {DATASTREAM}")
logging.debug(f"MCList: {MCList}")
logging.debug(f"SYSTs: {SYSTs}")

#### Get Histograms
HISTs = {}
COLORs = {}

#### helper functions
def hadd(file_path):
    if not os.path.exists(file_path):
        logging.info(os.listdir(f"{os.path.dirname(file_path)}"))
        logging.info("Hadding...")        
        os.system(f"hadd -f {file_path} {os.path.dirname(file_path)}/*")

def addHist(name, hist, histDict):
    if histDict[name] is None:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

## data
file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSelector/{args.era}/{SKIMFLAG}__/DATA/PromptSelector_{DATASTREAM}.root"
if "ZGamma" in args.channel:
    file_path = file_path.replace("PromptSelector", "MeasConversionV3")
logging.debug(f"file_path: {file_path}")
hadd(file_path)
assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile.Open(file_path)
try:
    data = f.Get(f"{args.channel}/Central/{args.histkey}"); data.SetDirectory(0)
except:
    logging.info(f"[WARNING] No data histogram found for {args.histkey}")
    exit(1)

# blind data
if args.blind:
    for bin in range(data.GetNcells()):
        data.SetBinContent(bin, 0)
        data.SetBinError(bin, 0)
f.Close()

## signals
if args.blind:
    SIGs = {}
    for signal in SIGNALs:
        file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSelector/{args.era}/{SKIMFLAG}__RunSyst__/PromptSelector_TTToHcToWAToMuMu_{signal}.root"
        logging.debug(f"file_path: {file_path}")
        assert os.path.exists(file_path), f"File not found: {file_path}"
        f = ROOT.TFile.Open(file_path)
        h = f.Get(f"{args.channel}/Central/{args.histkey}")
        if "score" in args.histkey: 
            h.Scale(5)
        SIGs[signal] = h.Clone(signal)
        SIGs[signal].SetDirectory(0)
        f.Close()

## nonprompt
file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MatrixSelector/{args.era}/{SKIMFLAG}__/DATA/MatrixSelector_{DATASTREAM}.root"
if "ZGamma" in args.channel:
    file_path = file_path.replace("MatrixSelector", "MeasConvMatrixV3")
logging.debug(f"file_path: {file_path}")
hadd(file_path)
assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile.Open(file_path)
nonprompt = f.Get(f"{args.channel}/Central/{args.histkey}"); nonprompt.SetDirectory(0)
for bin in range(nonprompt.GetNcells()):
    nonprompt.SetBinError(bin, nonprompt.GetBinContent(bin)*0.3)
f.Close()
HISTs["nonprompt"] = nonprompt.Clone("nonprompt")

for sample in MCList:
    file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSelector/{args.era}/{SKIMFLAG}__RunSyst__/PromptSelector_{sample}.root"
    if "ZGamma" in args.channel:
        file_path = file_path.replace("PromptSelector", "MeasConversionV3").replace("RunSyst__", "")
    logging.debug(f"file_path: {file_path}")
    assert os.path.exists(file_path), f"File not found: {file_path}"
    f = ROOT.TFile.Open(file_path)
    # Get Central Histograms
    try:
        h = f.Get(f"{args.channel}/Central/{args.histkey}"); h.SetDirectory(0)
        if sample in CONV:
            h.Scale(ConvSF)
    except Exception as e:
        logging.debug(e, sample)
        continue

    # Get Systematic Histograms
    hSysts = []
    for syst_up, syst_down in SYSTs:
        try:
            h_up = f.Get(f"{args.channel}/{syst_up}/{args.histkey}"); h_up.SetDirectory(0)
            h_down = f.Get(f"{args.channel}/{syst_down}/{args.histkey}"); h_down.SetDirectory(0)
            hSysts.append((h_up, h_down))
        except Exception as e:
            logging.debug(e, sample, syst_up, syst_down)
            continue
    f.Close()

    # estimate total unc. bin by bin
    for bin in range(h.GetNcells()):
        stat_unc = h.GetBinError(bin)
        if sample in CONV:
            total_unc = sqrt(pow(stat_unc, 2) + pow(h.GetBinContent(bin)*ConvSFErr, 2))
        else:
            envelops = []
            for h_up, h_down in hSysts:
                systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
                systDown = abs(h_down.GetBinContent(bin) - h.GetBinContent(bin))
                envelops.append(max(systUp, systDown))
            total_unc = sqrt(pow(stat_unc, 2) + sum([pow(x, 2) for x in envelops]))
        h.SetBinError(bin, total_unc)
    HISTs[sample] = h.Clone(sample)

temp_dict = {}
temp_dict["nonprompt"] = None
temp_dict["conversion"] = None
temp_dict["ttX"] = None
temp_dict["diboson"] = None
temp_dict["others"] = None

addHist("nonprompt", HISTs["nonprompt"], temp_dict)
for sample in CONV:
    if sample not in HISTs.keys(): continue
    addHist("conversion", HISTs[sample], temp_dict)
for sample in TTX:
    if sample not in HISTs.keys(): continue
    addHist("ttX", HISTs[sample], temp_dict)
for sample in DIBOSON:
    if sample not in HISTs.keys(): continue
    addHist("diboson", HISTs[sample], temp_dict)
for sample in OTHERs:
    if sample not in HISTs.keys(): continue
    addHist("others", HISTs[sample], temp_dict)

# filter out none historgrams from temp_dict
BKGs = {name: hist for name, hist in temp_dict.items() if hist}
logging.debug(f"BKGs: {BKGs}")

COLORs["data"] = ROOT.kBlack
COLORs["nonprompt"] = ROOT.kGray+2
COLORs["ttX"] = ROOT.kBlue
COLORs["conversion"] = ROOT.kViolet
COLORs["diboson"] = ROOT.kGreen
COLORs["others"] = ROOT.kAzure

COLORs["MHc-70_MA-15"] = ROOT.kGreen
COLORs["MHc-100_MA-60"] = ROOT.kBlue
COLORs["MHc-100_MA-95"] = ROOT.kViolet
COLORs["MHc-130_MA-90"] = ROOT.kBlack
COLORs["MHc-160_MA-85"] = ROOT.kBlue
COLORs["MHc-160_MA-155"] = ROOT.kRed

c = ComparisonCanvas(config=config)
if args.blind:
    c.drawSignals(SIGs, COLORs)
c.drawBackgrounds(BKGs, COLORs)
c.drawData(data)
c.drawRatio()
c.drawLegend()
c.finalize()

hist_path = f"{WORKDIR}/TriLepton/plots/{args.era}/{args.channel}/{args.histkey.replace('/', '_')}.png"
if not os.path.exists(os.path.dirname(hist_path)):
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
c.SaveAs(hist_path)
