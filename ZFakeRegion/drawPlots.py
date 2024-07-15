import os, sys; sys.path.append("/home/choij/workspace/ChargedHiggsAnalysisV2/CommonTools/python")
import argparse
import ROOT
import json 
from math import pow, sqrt
from plotter import ComparisonCanvas

# No graphics
ROOT.gROOT.SetBatch(True)

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--key", required=True, type=str, help="histkey")
parser.add_argument("--channel", required=True, type=str, help="channel")
args = parser.parse_args()

# read config
config_key = args.key
if args.key == "nonprompt/pt":
    if "1E2Mu" in args.channel: config_key = "electrons/1/pt"
    if "3Mu" in args.channel: config_key = "muons/1/pt"
if args.key == "nonprompt/eta":
    if "1E2Mu" in args.channel: config_key = "electrons/1/eta"
    if "3Mu" in args.channel: config_key = "muons/1/eta"
with open("histConfigs.json") as f:
    config = json.load(f)[config_key]
config["era"] = args.era
    
WORKDIR = os.environ["WORKDIR"]

## Sample List
DataStream = ""
if "1E2Mu" in args.channel: DataStream = "MuonEG"
if "3Mu" in args.channel: DataStream = "DoubleMuon"



CONV = ["DYJets_MG"]
#VV = ["WZTo3LNu_mllmin4p0_powheg", "ZZTo4L_powheg"]
VV = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
TTX = ["ttWToLNu", "ttZToLLNuNu", "ttHToNonbb"]
RARE = ["WWW", "WWZ", "WZZ", "ZZZ", "WWG", "TTG", "tZq", "tHq", "TTTT", "VBF_HToZZTo4L", "GluGluHToZZTo4L"]
MCSamples = CONV + VV + TTX + RARE

#### Systematics
if args.channel == "Skim1E2Mu":
    SYSTEMATICs = [["L1PrefireUp", "L1PrefireDown"],
                   ["PileupReweightUp", "PileupReweightDown"],
                   ["MuonIDSFUp", "MuonIDSFDown"],
                   ["ElectronIDSFUp", "ElectronIDSFDown"],
                   ["EMuTrigSFUp", "EMuTrigSFDown"],
                   ["JetEnUp", "JetEnDown"],
                   ["JetResUp", "JetResDown"],
                   ["ElectronResUp", "ElectronResDown"],
                   ["ElectronEnUp", "ElectronEnDown"],
                   ["MuonEnUp", "MuonEnDown"],
                   ["HeavyTagUpUnCorr", "HeavyTagDownUnCorr"],
                   ["HeavyTagUpCorr", "HeavyTagDownCorr"],
                   ["LightTagUpUnCorr", "LightTagDownUnCorr"],
                   ["LightTagUpCorr", "LightTagDownCorr"]]
elif args.channel == "Skim3Mu":
    SYSTEMATICs = [["L1PrefireUp", "L1PrefireDown"],
                   ["PileupReweightUp", "PileupReweightDown"],
                   ["MuonIDSFUp", "MuonIDSFDown"],
                   ["DblMuTrigSFUp", "DblMuTrigSFDown"],
                   ["JetEnUp", "JetEnDown"],
                   ["JetResUp", "JetResDown"],
                   ["MuonEnUp", "MuonEnDown"],
                   ["HeavyTagUpUnCorr", "HeavyTagDownUnCorr"],
                   ["HeavyTagUpCorr", "HeavyTagDownCorr"],
                   ["LightTagUpUnCorr", "LightTagDownUnCorr"],
                   ["LightTagUpCorr", "LightTagDownCorr"]]
else:
    raise ValueError(f"Invalid channel {argss.channel}")

#### get histograms
HISTs = {}
COLORs = {}

## data
fstring = f"{WORKDIR}/SKFlatOutput/PromptEstimator/{args.era}/{args.channel}__/DATA/PromptEstimator_{DataStream}.root"
assert os.path.exists(fstring)
f = ROOT.TFile.Open(fstring)
data = f.Get(f"{args.channel.replace('Skim', 'ZFake')}/Central/{args.key}"); data.SetDirectory(0)
f.Close()

## fake
fstring = f"{WORKDIR}/SKFlatOutput/MatrixEstimator/{args.era}/{args.channel}__RunSyst__/DATA/MatrixEstimator_{DataStream}.root"
assert os.path.exists(fstring)
f = ROOT.TFile.Open(fstring)
fake = f.Get(f"{args.channel.replace('Skim', 'ZFake')}/Central/{args.key}"); fake.SetDirectory(0)
for bin in range(fake.GetNcells()):
    fake.SetBinError(bin, fake.GetBinContent(bin)*0.3)
HISTs["nonprompt"] = fake

## prompt
for sample in MCSamples:
    fstring = f"{WORKDIR}/SKFlatOutput/PromptEstimator/{args.era}/{args.channel}__RunSyst__/PromptEstimator_{sample}.root"
    try:
        assert os.path.exists(fstring)
        f = ROOT.TFile.Open(fstring)
        h = f.Get(f"{args.channel.replace('Skim', 'ZFake')}/Central/{args.key}")
        h.SetDirectory(0)
        
        # get systematic histograms
        hSysts = []
        for syst in SYSTEMATICs:
            h_up = f.Get(f"{args.channel.replace('Skim', 'ZFake')}/{syst[0]}/{args.key}"); h_up.SetDirectory(0)
            h_down = f.Get(f"{args.channel.replace('Skim', 'ZFake')}/{syst[1]}/{args.key}"); h_down.SetDirectory(0)
            hSysts.append([h_up, h_down])

        for bin in range(h.GetNcells()):
            errInThisBin = [h.GetBinError(bin)]
            for hists in hSysts:
                systUp = abs(h_up.GetBinContent(bin)-h.GetBinContent(bin))
                systDown = abs(h_down.GetBinContent(bin)-h.GetBinContent(bin))
                errInThisBin.append(max(systUp, systDown))
            
            thisError = 0.
            for err in errInThisBin:
                thisError += pow(err, 2)
            thisError = sqrt(err)
            h.SetBinError(bin, thisError)
        f.Close()
        HISTs[sample] = h.Clone(sample)
    except Exception as e:
        print(sample, e)
        
#### merge backgrounds
def addHist(name, hist, histDict):
    if histDict[name] is None:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)
        
temp_dict = {}
temp_dict["nonprompt"] = None
temp_dict["conversion"] = None
temp_dict["ttX"] = None
temp_dict["diboson"] = None
temp_dict["others"] = None

addHist("nonprompt", HISTs['nonprompt'], temp_dict)
for sample in CONV:
    if not sample in HISTs.keys(): continue
    addHist("conversion", HISTs[sample], temp_dict)
for sample in TTX:
    if not sample in HISTs.keys(): continue
    addHist("ttX", HISTs[sample], temp_dict)
for sample in VV:
    if not sample in HISTs.keys(): continue
    addHist("diboson", HISTs[sample], temp_dict)
for sample in RARE:
    if not sample in HISTs.keys(): continue
    addHist("others", HISTs[sample], temp_dict)
    

#### remove none histogram
BKGs = {}
for key, value in temp_dict.items():
    if temp_dict[key] is None: continue
    BKGs[key] = value

colorList = [ROOT.kGray, ROOT.kGreen, ROOT.kCyan, ROOT.kBlue, ROOT.kBlack, ROOT.kInvertedDarkBodyRadiator, ROOT.kRed, ROOT.kOrange]

COLORs["data"] = ROOT.kBlack
COLORs['nonprompt'] = ROOT.kGray+2
COLORs["ttX"] = ROOT.kBlue
COLORs["conversion"] = ROOT.kViolet
COLORs["diboson"] = ROOT.kGreen
COLORs["others"] = ROOT.kAzure

c = ComparisonCanvas(config=config)
c.drawBackgrounds(BKGs, COLORs)
c.drawData(data)
c.drawRatio()
c.drawLegend()
c.finalize()

histpath = f"{WORKDIR}/ZFakeRegion/plots/{args.era}/{args.channel}/{args.key.replace('/', '_')}.png"
if not os.path.exists(os.path.dirname(histpath)):
    os.makedirs(os.path.dirname(histpath))
c.SaveAs(histpath)
