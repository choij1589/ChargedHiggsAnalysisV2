#!/usr/bin/env python
import os
import logging
import argparse
import json
import pandas as pd
import ROOT
from math import pow, sqrt
from ctypes import c_double
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="Skim1E2Mu / Skim3Mu")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']
histkey = "ZCand/mass"
logging.basicConfig(level=logging.DEBUG) if args.debug else logging.basicConfig(level=logging.INFO)

# Channel Dependent Settings
if args.channel == "Skim1E2Mu":
    DATASTREAM = "MuonEG"
    REGION = "ZGamma1E2Mu"
    SYSTs = [["L1PrefireUp", "L1PrefireDown"],
             ["PileupReweightUp", "PileupReweightDown"],
             ["MuonIDSFUp", "MuonIDSFDown"],
             ["ElectronIDSFUp", "ElectronIDSFDown"],
             ["EMuTrigSFUp", "EMuTrigSFDown"],
             ["JetResUp", "JetResDown"],
             ["JetEnUp", "JetEnDown"],
             ["ElectronResUp", "ElectronResDown"],
             ["ElectronEnUp", "ElectronEnDown"],
             ["MuonEnUp", "MuonEnDown"]]
elif args.channel == "Skim3Mu":
    DATASTREAM = "DoubleMuon"
    REGION = "ZGamma3Mu"
    SYSTs = [["L1PrefireUp", "L1PrefireDown"],
             ["PileupReweightUp", "PileupReweightDown"],
             ["MuonIDSFUp", "MuonIDSFDown"],
             ["DblMuTrigSFUp", "DblMuTrigSFDown"],
             ["JetResUp", "JetResDown"],
             ["JetEnUp", "JetEnDown"],
             ["ElectronResUp", "ElectronResDown"],
             ["ElectronEnUp", "ElectronEnDown"],
             ["MuonEnUp", "MuonEnDown"]]
else:
    raise ValueError(f"Invalid channel {args.channel}")

# Set Sample Definitions
CONV = ["DYJets", "DYJets10to50_MG", "ZGToLLG", "TTG"] # DYJets10to50_MG and TTG Contribution is negligible
DIBOSON = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
TTX = ["ttWToLNu", "ttZToLLNuNu", "ttHToNonbb"]
RARE    = ["WWW", "WWZ", "WZZ", "ZZZ", "tZq", "tHq", "TTTT", "WWG", "VBF_HToZZTo4L", "GluGluHToZZTo4L"]
MCLists = CONV + DIBOSON + TTX + RARE
PromptBkgs = DIBOSON + TTX + RARE

# Helper Functions
def hadd(file_path):
    print(f"file {file_path} does not exist. hadding...")
    os.system(f"hadd -f {file_path} {file_path.replace('.root', '_*.root')}")

def extract_data_from_hist(sample):
    data = {}
    
    # Open root file
    if sample == DATASTREAM:
        file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasConversionV3/{args.era}/{args.channel}__/DATA/MeasConversionV3_{DATASTREAM}.root"
        if not os.path.exists(file_path):
            hadd(file_path)
    elif sample == "nonprompt":
        file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasConvMatrixV3/{args.era}/{args.channel}__/DATA/MeasConvMatrixV3_{DATASTREAM}.root"
        if not os.path.exists(file_path):
            hadd(file_path)
    else:
        file_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasConversionV3/{args.era}/{args.channel}__/MeasConversionV3_{sample}.root"
    assert os.path.exists(file_path), f"file {file_path} does not exist"
    f = ROOT.TFile.Open(file_path)
    try:
        h = f.Get(f"{REGION}/Central/{histkey}"); h.SetDirectory(0)
    
        # Extract rate and stat error
        stat = c_double()
        rate = h.IntegralAndError(0, h.GetNbinsX()+1, stat)
    
        data["Central"] = rate
        data["Stat"] = stat.value
    except:
        logging.warning(f"Failed to extract Central for {sample}")
        data["Central"] = None
        data["Stat"] = None
    
    # Now extract systematics
    if sample not in [DATASTREAM, "nonprompt"]:
        for syst in SYSTs:
            for s in syst:
                try:
                    h = f.Get(f"{REGION}/{s}/{histkey}"); h.SetDirectory(0)
                    rate = h.Integral()
                    data[s] = rate
                except:
                    logging.warning(f"Failed to extract {s} for {sample}")
                    data[s] = None
    f.Close()
    
    return data

def save_data_to_json(data, output_file):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def get_value(df, sample_name, syst):
    return df.filter(like=sample_name, axis=1)[f"{sample_name}.{syst}"].values[0]
        
def estimate_total_err(df, sample_name):
    try:
        central = get_value(df, sample_name, "Central")
        total = pow(get_value(df, sample_name, "Stat"), 2)
    except:
        logging.warning(f"Failed to estimate total error for {sample_name}")
        return None
    
    for syst in SYSTs:
        syst_up = abs(get_value(df, sample_name, syst[0]) - central)
        syst_down = abs(get_value(df, sample_name, syst[1]) - central)
        total += pow(max(syst_up, syst_down), 2)
    
    return sqrt(total)

if __name__ == "__main__":
    # Extract integrals and make json file
    data = {}
    data[DATASTREAM] = extract_data_from_hist(DATASTREAM)
    data["nonprompt"] = extract_data_from_hist("nonprompt")
    for sample in MCLists:
        data[sample] = extract_data_from_hist(sample)
    json_output_path = f"{WORKDIR}/MeasConversion/results/{args.era}/json/{args.channel}.json"
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    save_data_to_json(data, json_output_path)
    
    # Reload json file and make flat dataframe
    with open(json_output_path, "r") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    
    # Save with TH1 format
    RatesWithError = ROOT.TH1F("RatesWithError", "RatesWithError", 6, 0, 6)

    # DATA
    RatesWithError.GetXaxis().SetBinLabel(1, "DATA")
    RatesWithError.SetBinContent(1, df.loc[0, f"{DATASTREAM}.Central"])
    RatesWithError.SetBinError(1, df.loc[0, f"{DATASTREAM}.Stat"])

    # Nonprompt
    RatesWithError.GetXaxis().SetBinLabel(2, "nonprompt")
    RatesWithError.SetBinContent(2, df.loc[0, "nonprompt.Central"])
    err = sqrt(pow(df.loc[0, "nonprompt.Stat"], 2) + pow(df.loc[0, "nonprompt.Central"]*0.3, 2))
    RatesWithError.SetBinError(2, err)

    # Conversion
    rate_conv = 0
    err_conv = 0
    for sample in CONV:
        try:
            rate_conv += df.loc[0, f"{sample}.Central"]
            err_conv += estimate_total_err(df, sample)
        except:
            logging.warning(f"Failed to extract {sample}")
    RatesWithError.GetXaxis().SetBinLabel(3, "Conversion")
    RatesWithError.SetBinContent(3, rate_conv)
    RatesWithError.SetBinError(3, err_conv)

    # Prompt backgrounds
    # Diboson
    rate_diboson = 0
    err_diboson = 0
    for sample in DIBOSON:
        rate_diboson += df.loc[0, f"{sample}.Central"]
        err_diboson += estimate_total_err(df, sample)
    RatesWithError.GetXaxis().SetBinLabel(4, "VV")
    RatesWithError.SetBinContent(4, rate_diboson)
    RatesWithError.SetBinError(4, err_diboson)

    # ttX
    rate_ttX = 0
    err_ttX = 0
    for sample in TTX:
        rate_ttX += df.loc[0, f"{sample}.Central"]
        err_ttX += estimate_total_err(df, sample)
    RatesWithError.GetXaxis().SetBinLabel(5, "ttX")
    RatesWithError.SetBinContent(5, rate_ttX)
    RatesWithError.SetBinError(5, err_ttX)

    # rare
    rate_rare = 0
    err_rare = 0
    for sample in RARE:
        try:
            rate_rare += df.loc[0, f"{sample}.Central"]
            err_rare += estimate_total_err(df, sample)
        except:
            logging.warning(f"Failed to extract {sample}")
    RatesWithError.GetXaxis().SetBinLabel(6, "rare")
    RatesWithError.SetBinContent(6, rate_rare)
    RatesWithError.SetBinError(6, err_rare)
    
    # Estimate Conversion Scale Factor
    rate_data = RatesWithError.GetBinContent(1)
    rate_conv = RatesWithError.GetBinContent(3)
    rate_pred = RatesWithError.GetBinContent(2) + RatesWithError.GetBinContent(4) + RatesWithError.GetBinContent(5) + RatesWithError.GetBinContent(6)

    dsf_ddata = 1 / rate_conv
    dsf_dconv = - (rate_data - rate_pred) / pow(rate_conv, 2)
    dsf_dpred = -1 / rate_conv

    sf = (rate_data - rate_pred) / rate_conv
    sf_err = sqrt(pow(dsf_ddata*RatesWithError.GetBinError(1), 2) + pow(dsf_dconv*RatesWithError.GetBinError(3), 2) + pow(dsf_dpred*(RatesWithError.GetBinError(2) + RatesWithError.GetBinError(4) + RatesWithError.GetBinError(5) + RatesWithError.GetBinError(6)), 2))

    # Not draw plots
    RatesWithError.SetStats(0)
    RatesWithError.SetTitle(f"{args.era}-{REGION}")
    RatesWithError.SetMarkerStyle(20)
    RatesWithError.SetLineColor(ROOT.kBlack)
    RatesWithError.GetXaxis().SetLabelSize(0.04)
    RatesWithError.GetXaxis().SetLabelFont(42)
    
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.04)
    
    c = ROOT.TCanvas("c", "c", 700, 800)
    c.cd()
    RatesWithError.Draw("PE")
    text.DrawLatex(0.65, 0.7, f"{sf:.3f} #pm {sf_err:.3f}")
    c.RedrawAxis()
    c.SaveAs(f"{WORKDIR}/MeasConversion/results/{args.era}/{args.channel}.png")
    
    