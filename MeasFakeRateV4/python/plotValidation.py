#!/home/choij/miniconda3/envs/pyg/bin/python
import os, sys; sys.path.append("/home/choij/workspace/ChargedHiggsAnalysisV2/CommonTools/python")
import argparse
import ROOT
import pandas as pd
from math import pow, sqrt
from itertools import product
from plotter import ComparisonCanvas

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hlt", required=True, type=str, help="hlt")
parser.add_argument("--wp", required=True, type=str, help="wp")
parser.add_argument("--region", required=True, type=str, help="region")
parser.add_argument("--syst", default="Central", type=str, help="systematic")
parser.add_argument("--full", default=False, action="store_true", help="full systematics")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']
if "El" in args.hlt:
    MEASURE = "electron"
    if args.hlt == "MeasFakeEl8":
        ptcorr_bins = [15., 20., 25., 35., 50., 100.]
    elif args.hlt == "MeasFakeEl12":
        ptcorr_bins = [15., 20., 25., 35., 50., 100.]
    elif args.hlt == "MeasFakeEl23":
        ptcorr_bins = [25., 35., 50., 100.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.8, 1.479, 2.5]
elif "Mu" in args.hlt:
    MEASURE = "muon"
    if args.hlt == "MeasFakeMu8":
        ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    elif args.hlt == "MeasFakeMu17":
        ptcorr_bins = [20., 30., 50., 100.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.9, 1.6, 2.4]
else:
    raise ValueError(f"invalid hlt {args.hlt}")

def findbin(ptcorr, abseta):
    if ptcorr > 200.:
        ptcorr = 199.
        
    prefix = ""
    # find bin index for ptcorr
    for i, _ in enumerate(ptcorr_bins[:-1]):
        if ptcorr_bins[i] < ptcorr+0.1 < ptcorr_bins[i+1]:
            prefix += f"ptcorr_{int(ptcorr_bins[i])}to{int(ptcorr_bins[i+1])}"
            break
    
    # find bin index for abseta
    abseta_idx = -1
    for i, _ in enumerate(abseta_bins[:-1]):
        if abseta_bins[i] < abseta+0.001 < abseta_bins[i+1]:
            abseta_idx = i
            break
    
    if abseta_idx == 0:   prefix += "_EB1"
    elif abseta_idx == 1: prefix += "_EB2"
    elif abseta_idx == 2: prefix += "_EE"
    else:        raise ValueError(f"Wrong abseta {abseta}")
    
    return prefix

trigPathDict = {
    "MeasFakeMu8": "HLT_Mu8_TrkIsoVVL_v",
    "MeasFakeMu17": "HLT_Mu17_TrkIsoVVL_v",
    "MeasFakeEl8": "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v",
    "MeasFakeEl12": "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v",
    "MeasFakeEl23": "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v"
}

## sample lists
DataStream = ""
if "El" in args.hlt:
    if "2016" in args.era:  DataStream = "DoubleEG"
    if "2017" in args.era:  DataStream = "SingleElectron"
    if "2018" in args.era:  DataStream = "EGamma"
    QCD = ["QCD_EMEnriched", "QCD_bcToE"]
if "Mu" in args.hlt:
    DataStream = "DoubleMuon"
    QCD = ["QCD_MuEnriched"]
    
W  = ["WJets_MG"]
DY = ["DYJets", "DYJets10to50_MG"]
TT = ["TTLL_powheg", "TTLJ_powheg"]
VV = ["WW_pythia", "WZ_pythia", "ZZ_pythia"]
ST = ["SingleTop_sch_Lep", "SingleTop_tch_top_Incl", "SingleTop_tch_antitop_Incl",
      "SingleTop_tW_top_NoFullyHad", "SingleTop_tW_antitop_NoFullyHad"]
MCList = W + DY + TT + VV + ST

## Systemtatics
SYSTs = []
if "El" in args.hlt and args.full:
    SYSTs.append(("PileupReweight"))
    SYSTs.append(("L1PrefireUp", "L1PrefireDown"))
    SYSTs.append(("ElectronRecoSFUp", "ElectronRecoSFDown"))
    SYSTs.append(("JetResUp", "JetResDown"))
    SYSTs.append(("JetEnUp", "JetEnDown"))
    SYSTs.append(("ElectronResUp", "ElectronResDown"))
    SYSTs.append(("ElectronEnUp", "ElectronEnDown"))
    SYSTs.append(("MuonEnUp", "MuonEnDown"))
if "Mu" in args.hlt and args.full:
    SYSTs.append(("PileupReweight"))
    SYSTs.append(("L1PrefireUp", "L1PrefireDown"))
    SYSTs.append(("MuonRecoSFUp", "MuonRecoSFDown"))
    SYSTs.append(("JetResUp", "JetResDown"))
    SYSTs.append(("JetEnUp", "JetEnDown"))
    SYSTs.append(("ElectronResUp", "ElectronResDown"))
    SYSTs.append(("ElectronEnUp", "ElectronEnDown"))
    SYSTs.append(("MuonEnUp", "MuonEnDown"))

# start loop over ptcorr x abseta
for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
    prefix = findbin(ptcorr, abseta)
    ## get histograms
    HISTs = {}
    COLORs = {}

    # data
    file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{args.hlt}__RunSyst__/DATA/MeasFakeRateV4_{DataStream}.root"
    try:
        assert os.path.exists(file_path) 
    except:
        raise FileNotFoundError(f"{file_path} does not exist")
    f = ROOT.TFile.Open(file_path)
    data = f.Get(f"{prefix}/{args.region}/{args.wp}/{args.syst}/MT"); data.SetDirectory(0)
    f.Close()

    for sample in MCList+QCD:
        file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{args.hlt}__RunSyst__/MeasFakeRateV4_{sample}.root"
        # get central histogram
        try:
            assert os.path.exists(file_path)
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{prefix}/{args.region}/{args.wp}/{args.syst}/MT");   h.SetDirectory(0)
            # get systematic histograms
            hSysts = []
            for systset in SYSTs:
                if len(systset) == 2:
                    systUp, systDown = systset
                    h_up = f.Get(f"{args.region}/{args.wp}/{systUp}/MT"); h_up.SetDirectory(0)
                    h_down = f.Get(f"{args.region}/{args.wp}/{systDown}/MT"); h_down.SetDirectory(0) 
                    hSysts.append((h_up, h_down))
                else:
                    # only one systematic source
                    syst = systset
                    h_syst = f.Get(f"{args.region}/{args.wp}/{syst}/MT"); h_syst.SetDirectory(0)
                    hSysts.append((h_syst))
            f.Close()
        except Exception as e:
            print(e, sample)
            continue
    
        # estimate total unc. bin by bin
        for bin in range(h.GetNcells()):
            stat_unc = h.GetBinError(bin)
            envelops = []
            for hset in hSysts:
                if len(hset) == 2:
                    h_up, h_down = hset
                    systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
                    systDown = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
                    envelops.append(max(systUp, systDown))
                else:
                    h_syst = hset
                    syst = abs(h_syst.GetBinContent(bin) - h.GetBinContent(bin))
                    envelops.append(syst)

                total_unc = pow(stat_unc, 2)
                for unc in envelops:
                    total_unc += pow(unc, 2)
                total_unc = sqrt(total_unc)
                h.SetBinError(bin, total_unc)
        HISTs[sample] = h.Clone(sample)
    
    # now scale MC histograms
    # get normalization factors
    prompt_scales = pd.read_csv(f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/CSV/{MEASURE}/prompt_scale.csv", index_col=[0, 1, 2])
    prompt_scale = float(prompt_scales.loc[(args.hlt, args.wp, args.syst), "Scale"])

    for hist in HISTs.values():
        hist.Scale(prompt_scale)

    ## merge backgrounds
    def add_hist(name, hist, histDict):
        # content of dictionary should be initialized as "None"
        if histDict[name] is None:
            histDict[name] = hist.Clone(name)
        else:
            histDict[name].Add(hist)

    temp_dict = {}
    temp_dict["W"]  = None
    temp_dict["DY"] = None
    temp_dict["TT"] = None
    temp_dict["VV"] = None 
    temp_dict["ST"] = None
    temp_dict["QCD"] = None

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
    for sample in QCD:
        if not sample in HISTs.keys(): continue
        add_hist("QCD", HISTs[sample], temp_dict)

    ## remove none histograms
    BKGs = {}
    for key, value in temp_dict.items():
        if value is None: continue
        BKGs[key] = value

    COLORs["data"] = ROOT.kBlack
    COLORs["W"]  = ROOT.kMagenta
    COLORs["DY"] = ROOT.kBlue
    COLORs["TT"] = ROOT.kViolet
    COLORs["VV"] = ROOT.kGreen
    COLORs["ST"] = ROOT.kAzure
    COLORs["QCD"] = ROOT.kRed

    xTitle = ""
    if "Mu" in args.hlt:
        xTitle = "M_{T}"
        xRange = [0., 200.]
    if "El" in args.hlt:
        xTitle = "M_{T}"
        xRange = [0., 200.]

    config = {"era": args.era,
              "xTitle": xTitle,
              "yTitle": "Events / 5GeV",
              "xRange": xRange,
              "yRange": [0., 2.],
              "rebin": 5}

    textInfo = {
        "CMS": [0.04, 61, [0.12, 0.91]],
        "Work in progress": [0.035, 52, [0.21, 0.91]],
        "Prescaled (13 TeV)": [0.035, 42, [0.665, 0.91]],
        trigPathDict[args.hlt]: [0.035, 42, [0.17, 0.83]],
        f"{args.era} / {args.wp.upper()} ID": [0.035, 42, [0.17, 0.77]]
    }

    c = ComparisonCanvas(config=config)
    c.drawBackgrounds(BKGs, COLORs)
    c.drawData(data)
    c.drawRatio()
    c.drawLegend()
    c.finalize(textInfo=textInfo)

    output_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/plots/{MEASURE}/{args.region}/{args.syst}/{prefix}_{args.hlt}_{args.wp}_MT.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c.SaveAs(output_path)
