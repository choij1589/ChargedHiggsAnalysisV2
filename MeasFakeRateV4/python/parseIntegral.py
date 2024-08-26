#!/usr/bin/env python
import os
import argparse
import ROOT
import pandas as pd
from ctypes import c_double
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
args = parser.parse_args()

#### Settings
WORKDIR = os.environ['WORKDIR']
DATASTREAM = ""
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
else:
    raise KeyError(f"Wrong measure {args.measure}")

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

def get_hist(sample, ptcorr, abseta, wp, syst="Central"):
    prefix = findbin(ptcorr, abseta)
    channel = ""
    if args.measure == "muon":
        if ptcorr < 30.: channel = "MeasFakeMu8"
        else:            channel = "MeasFakeMu17"
    elif args.measure == "electron":
        #if ptcorr < 15.:   channel = "MeasFakeEl8"
        if ptcorr < 35.: channel = "MeasFakeEl12"
        else:              channel = "MeasFakeEl23"
    
    file_path = ""
    if sample == DATASTREAM:
        file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{channel}__RunSyst__/DATA/MeasFakeRateV4_{sample}.root"
    else:
        file_path = f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{channel}__RunSyst__/MeasFakeRateV4_{sample}.root"
        
    try:
        assert os.path.exists(file_path)
    except AssertionError:
        raise AssertionError(f"File not found: {file_path}")

    f = ROOT.TFile.Open(file_path)
    try:
        if syst == "Stat":
            h = f.Get(f"{prefix}/QCDEnriched/{wp}/Central/MT"); h.SetDirectory(0)
        else:
            h = f.Get(f"{prefix}/QCDEnriched/{wp}/{syst}/MT"); h.SetDirectory(0)
        f.Close()
        return h
    except:
        raise KeyError(f"Cannot find {prefix}/QCDEnriched/{wp}/{syst}/MT for sample {sample}")

def collect_data(sample, wp, syst):
    data = []
    for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
        h = get_hist(sample, ptcorr, abseta, wp, syst)
        if syst == "Stat":
            err = c_double()
            rate = h.IntegralAndError(0, h.GetNbinsX()+1, err)
            data.append(err.value)
        else:
            rate = h.Integral(0, h.GetNbinsX()+1)
            data.append(rate)
    return data

if __name__ == "__main__":
    # Make DataFrame
    index_col = []
    for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
        index_col.append(findbin(ptcorr, abseta))
        
    # Collect DATASTREAM
    data_dict = {}
    for syst in SYSTs:
        data = collect_data(DATASTREAM, "loose", syst)
        data_dict[syst] = data
    df = pd.DataFrame(data_dict, index=index_col)
    csv_path = f"results/{args.era}/CSV/{args.measure}/{DATASTREAM}_loose.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path)
    
    data_dict = {}
    for syst in SYSTs:
        data = collect_data(DATASTREAM, "tight", syst)
        data_dict[syst] = data
    df = pd.DataFrame(data_dict, index=index_col)
    csv_path = f"results/{args.era}/CSV/{args.measure}/{DATASTREAM}_tight.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path)
    
    # Collect MC
    for sample, wp in product(MCList, ["loose", "tight"]):
        data_dict = {}
        for syst in SYSTs:
            data = collect_data(sample, wp, syst)
            data_dict[syst] = data
        df = pd.DataFrame(data_dict, index=index_col)
        csv_path = f"results/{args.era}/CSV/{args.measure}/{sample}_{wp}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path)
