#!/usr/bin/env python
import os
import argparse
import logging
import pandas as pd
import ROOT
from itertools import product
from array import array

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--isQCD", default=False, action="store_true", help="isQCD")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug:
    logging.basicConfig(level=logging.DEBUG)

#### Settings
WORKDIR = os.environ['WORKDIR']
DATASTREAM = ""
HLTPATHs = []
ptcorr_bins = []
abseta_bins = []
SYSTs = []

if args.measure == "electron":
    if "2016" in args.era:  DATASTREAM = "DoubleEG"
    if "2017" in args.era:  DATASTREAM = "SingleElectron"
    if "2018" in args.era:  DATASTREAM = "EGamma"
    ptcorr_bins = [15., 20., 25., 35., 50., 100.]
    abseta_bins = [0., 0.8, 1.479, 2.5]
    QCD = ["QCD_EMEnriched", "QCD_bcToE"]
    #QCD = ["QCD"]
    SYSTs = ["Central", "Stat",
             #"PileupReweight",
             #"L1PrefireUp", "L1PrefireDown",
             #"ElectronRecoSFUp", "ElectronRecoSFDown",
             #"JetResUp", "JetResDown",
             #"JetEnUp", "JetEnDown",
             #"ElectronResUp", "ElectronResDown",
             #"ElectronEnUp", "ElectronEnDown",
             #"MuonEnUp", "MuonEnDown",
             "MotherJetPtUp", "MotherJetPtDown",
             "RequireHeavyTag"]
    HLTPATHs = ["MeasFakeEl12", "MeasFakeEl23"]
elif args.measure == "muon":
    DATASTREAM = "DoubleMuon"
    ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    abseta_bins = [0., 0.9, 1.6, 2.4]
    QCD = ["QCD_MuEnriched"]
    SYSTs = ["Central", "Stat",
             #"PileupReweight",
             #"L1PrefireUp", "L1PrefireDown",
             #"MuonRecoSFUp", "MuonRecoSFDown",
             #"JetResUp", "JetResDown",
             #"JetEnUp", "JetEnDown",
             #"ElectronResUp", "ElectronResDown",
             #"ElectronEnUp", "ElectronEnDown",
             #"MuonEnUp", "MuonEnDown",
             "MotherJetPtUp", "MotherJetPtDown",
             "RequireHeavyTag"]
    HLTPATHs = ["MeasFakeMu8", "MeasFakeMu17"]
else:
    raise KeyError(f"Wrong measure {args.measure}")

SelectionVariations = ["Central",
                       "PromptNormUp", "PromptNormDown",
                       "MotherJetPtUp", "MotherJetPtDown",
                       "RequireHeavyTag"]
WPs = ["loose", "tight"]

W  = ["WJets_MG"]
DY = ["DYJets", "DYJets10to50_MG"]
TT = ["TTLL_powheg", "TTLJ_powheg"]
VV = ["WW_pythia", "WZ_pythia", "ZZ_pythia"]
ST = ["SingleTop_sch_Lep", "SingleTop_tch_top_Incl", "SingleTop_tch_antitop_Incl",
      "SingleTop_tW_top_NoFullyHad", "SingleTop_tW_antitop_NoFullyHad"]
MCList = W + DY + TT + VV + ST

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

#### first evaluate central scale for product(hlt, wp, syst)
def get_prompt_scale(hltpath, wp, syst):
    file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{hltpath}__RunSyst__/DATA/MeasFakeRateV4_{DATASTREAM}.root"
    histkey = f"ZEnriched/{wp}/{syst}/ZCand/mass"
    if syst in ["PromptNormUp", "PromptNormDown"]:
        histkey = f"ZEnriched/{wp}/Central/ZCand/mass"
    if syst == "Stat": histkey = f"ZEnriched/{wp}/Central/ZCand/mass"
    try:
        assert os.path.exists(file_path)
    except AssertionError:
        raise FileNotFoundError(f"{file_path} does not exist")
    f = ROOT.TFile.Open(file_path)
    data = f.Get(histkey); data.SetDirectory(0)
    f.Close()
    
    rate_data = data.Integral(0, data.GetNbinsX()+1)
    
    rate_mc = 0.
    for sample in MCList:
        file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{hltpath}__RunSyst__/MeasFakeRateV4_{sample}.root"
        try:
            assert os.path.exists(file_path)
        except AssertionError:
            raise FileNotFoundError(f"{file_path} does not exist")
        f = ROOT.TFile.Open(file_path)
        try:
            hist = f.Get(histkey); hist.SetDirectory(0)
            rate_mc += hist.Integral(0, hist.GetNbinsX()+1)
        except Exception as e:
            logging.debug(f"WARNING - No events for {sample} in ZEnriched region")
            continue
        f.Close()
        
    scale = rate_data / rate_mc
    
    if syst == "PromptNormUp": scale *= 1.15
    if syst == "PromptNormDown": scale *= 0.85
    
    return scale

#### Collect scales
#### For prompt normalization, assign 10% scale variation
scale_dict = {}
for hltpath, wp, syst in product(HLTPATHs, WPs, SelectionVariations):
    scale = get_prompt_scale(hltpath, wp, syst)
    scale_dict[(hltpath, wp, syst)] = scale

#### Save scales
df = pd.DataFrame(scale_dict, index=["Scale"]).T
csv_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/CSV/{args.measure}/prompt_scale.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path)

def get_nonprompt_from_data(ptcorr, abseta, wp, syst):
    prefix = findbin(ptcorr, abseta)
        
    # get integral for data
    csv_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/CSV/{args.measure}/{DATASTREAM}_{wp}.csv"
    df = pd.read_csv(csv_path, index_col=0)
    if syst in ["PromptNormUp", "PromptNormDown"]:
        data = df.loc[prefix, "Central"]
    else:
        data = df.loc[prefix, syst]
        
    # get MCs
    prompt = 0.
    for sample in MCList:
        csv_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/CSV/{args.measure}/{sample}_{wp}.csv"
        df = pd.read_csv(csv_path, index_col=0)
        if syst in ["PromptNormUp", "PromptNormDown"]:
            prompt += df.loc[prefix, "Central"]
        else:
            prompt += df.loc[prefix, syst]
        
    # get prompt scale
    if args.measure == "electron":
        if ptcorr < 35.: hltpath = "MeasFakeEl12"
        else:            hltpath = "MeasFakeEl23"
    elif args.measure == "muon":
        if ptcorr < 30.: hltpath = "MeasFakeMu8"
        else:            hltpath = "MeasFakeMu17"
    else:
        raise KeyError(f"Wrong measure {args.measure}")
    scale = get_prompt_scale(hltpath, wp, syst)
    logging.debug(prefix, data, prompt, scale, prompt*scale)
    
    return data - prompt*scale

def get_nonprompt_from_qcd(ptcorr, abseta, wp, sample):
    prefix = findbin(ptcorr, abseta)
        
    # get integral from QCD
    if args.measure == "electron":
        if ptcorr < 35.: hltpath = "MeasFakeEl12"
        else:            hltpath = "MeasFakeEl23"
        file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{hltpath}__RunSyst__/MeasFakeRateV4_{sample}.root"
        histkey = f"{prefix}/Inclusive/{wp}/Central/MT"
        
        f = ROOT.TFile.Open(file_path)
        h = f.Get(histkey); h.SetDirectory(0)
        f.Close()
               
        return h.Integral()
    else: # muon
        if ptcorr < 30.: hltpath = "MeasFakeMu8"
        else:            hltpath = "MeasFakeMu17"
        file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{hltpath}__RunSyst__/MeasFakeRateV4_{sample}.root"
        histkey = f"{prefix}/Inclusive/{wp}/Central/MT"

        f = ROOT.TFile.Open(file_path)
        h = f.Get(histkey); h.SetDirectory(0)
        f.Close()

        return h.Integral()
    

def get_fake_rate(isQCD=False, syst="Central"):
    h_loose = ROOT.TH2D("h_loose", "h_loose", len(abseta_bins)-1, array('d', abseta_bins), len(ptcorr_bins)-1, array('d', ptcorr_bins))
    h_tight = ROOT.TH2D("h_tight", "h_tight", len(abseta_bins)-1, array('d', abseta_bins), len(ptcorr_bins)-1, array('d', ptcorr_bins))
    
    if args.measure == "electron":
        sample = "QCD_EMEnriched"
    elif args.measure == "muon":
        sample = "QCD_MuEnriched"
    else:
        raise KeyError(f"Wrong measure {args.measure}")

    for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
        if isQCD:
            loose = get_nonprompt_from_qcd(ptcorr, abseta, "loose", sample)
            tight = get_nonprompt_from_qcd(ptcorr, abseta, "tight", sample)
        else:
            loose = get_nonprompt_from_data(ptcorr, abseta, "loose", syst)
            tight = get_nonprompt_from_data(ptcorr, abseta, "tight", syst)
        
        h_loose.Fill(abseta, ptcorr, loose)
        h_tight.Fill(abseta, ptcorr, tight)
    
    fake_rate = h_tight.Clone(f"fake rate - ({syst})")
    fake_rate.Divide(h_loose)
    fake_rate.SetTitle(f"fake rate - ({syst})")
    fake_rate.SetDirectory(0)
    return fake_rate

if __name__ == "__main__":
    output_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate.root"
    if args.isQCD:
        output_path = output_path.replace("fakerate", "fakerate_qcd")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = ROOT.TFile.Open(output_path, "RECREATE")
    if args.isQCD:
        for sample in QCD:
            h = get_fake_rate(args.isQCD, sample)
            out.cd()
            h.Write()
    else:
        for syst in SelectionVariations:
            h = get_fake_rate(args.isQCD, syst)
            out.cd()
            h.Write()
    out.Close()
