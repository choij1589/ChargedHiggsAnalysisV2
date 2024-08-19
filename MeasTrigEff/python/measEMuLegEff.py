#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
from math import pow, sqrt

WORKDIR = os.environ['WORKDIR']
parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hltpath", required=True, type=str, help="Mu8El23 / Mu23El12")
parser.add_argument("--leg", required=True, type=str, help="muon / electron")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug: logging.basicConfig(level=logging.DEBUG)

if args.leg == "muon":
    DATASTREAM = "EGamma" if args.era == "2018" else "SingleElectron"
    FLAG = "MeasMuLegs"
    LEG = "MuLeg"
elif args.leg == "electron":
    DATASTREAM = "SingleMuon"
    FLAG = "MeasElLegs"
    LEG = "ElLeg"
else:
    raise ValueError(f"Invalid leg {args.leg}")

# helper functions
def get_histograms(hltpath: ROOT.TString, syst: ROOT.TString, is_data: bool) -> (ROOT.TH2D, ROOT.TH2D):
    if is_data:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/DATA/MeasTrigEff_{DATASTREAM}.root")
    elif syst == "AltMC":
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/MeasTrigEff_DYJets.root")
    else:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/MeasTrigEff_TTLL_powheg.root")
        
    if syst == "AltTag":
        h_denom = f.Get(f"TrigEff_{hltpath}_{LEG}_DENOM/{syst}/fEta_Pt"); h_denom.SetDirectory(0)
        h_num = f.Get(f"TrigEff_{hltpath}_{LEG}_NUM/{syst}/fEta_Pt"); h_num.SetDirectory(0)
    else:
        h_denom = f.Get(f"TrigEff_{hltpath}_{LEG}_DENOM/Central/fEta_Pt"); h_denom.SetDirectory(0)
        h_num = f.Get(f"TrigEff_{hltpath}_{LEG}_NUM/Central/fEta_Pt"); h_num.SetDirectory(0)
    f.Close()
    
    return (h_num, h_denom)

def get_efficiency(histkey: ROOT.TString):
    hltpath, _, syst = tuple(histkey.split("_"))
    is_data = True if "Data" in histkey else False
    h_num, h_denom = get_histograms(hltpath, syst, is_data)
    h_eff = h_num.Clone(f"Eff_{histkey}"); h_eff.Reset()
    for bin in range(1, h_num.GetNcells()+1):
        num, err_num = h_num.GetBinContent(bin), h_num.GetBinError(bin)
        denom, err_denom = h_denom.GetBinContent(bin), h_denom.GetBinError(bin)
        if denom == 0 or num == 0:
            eff, err = 0., 0.
        else:
            eff = num / denom
            err = eff*sqrt(pow(err_num/num, 2) + pow(err_denom/denom, 2))
        h_eff.SetBinContent(bin, eff)
        h_eff.SetBinError(bin, err)
    h_eff.SetDirectory(0) 
    return h_eff

def main():
    # Get Efficiency Histograms
    h_eff_data_central = get_efficiency(f"{args.hltpath}_Data_Central")
    h_eff_data_alttag = get_efficiency(f"{args.hltpath}_Data_AltTag")
    h_eff_mc_central = get_efficiency(f"{args.hltpath}_MC_Central")
    h_eff_mc_altmc = get_efficiency(f"{args.hltpath}_MC_AltMC")
    h_eff_mc_alttag = get_efficiency(f"{args.hltpath}_MC_AltTag")
    
    # Calculate Systematic Uncertainties
    h_eff_data = h_eff_data_central.Clone(f"{args.hltpath}_Data"); h_eff_data.Reset()
    for bin in range(1, h_eff_data.GetNcells()+1):
        stat = h_eff_data_central.GetBinError(bin)
        diff_alttag = h_eff_data_alttag.GetBinContent(bin) - h_eff_data_central.GetBinContent(bin)
        total = sqrt(pow(stat, 2) + pow(diff_alttag, 2))
        h_eff_data.SetBinContent(bin, h_eff_data_central.GetBinContent(bin))
        h_eff_data.SetBinError(bin, total)
    
    h_eff_mc = h_eff_mc_central.Clone(f"{args.hltpath}_MC"); h_eff_mc.Reset()
    for bin in range(1, h_eff_mc.GetNcells()+1):
        stat = h_eff_mc_central.GetBinError(bin)
        diff_altmc = h_eff_mc_altmc.GetBinContent(bin) - h_eff_mc_central.GetBinContent(bin)
        diff_alttag = h_eff_mc_alttag.GetBinContent(bin) - h_eff_mc_central.GetBinContent(bin)
        total = sqrt(pow(stat, 2) + pow(diff_altmc, 2) + pow(diff_alttag, 2))
        h_eff_mc.SetBinContent(bin, h_eff_mc_central.GetBinContent(bin))
        h_eff_mc.SetBinError(bin, total)
        
    # Save Histograms
    outpath = f"{WORKDIR}/MeasTrigEff/results/{args.era}/{args.hltpath}_{args.leg}.root"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    f = ROOT.TFile(outpath, "RECREATE")
    f.cd()
    h_eff_data_central.Write()
    h_eff_data_alttag.Write()
    h_eff_mc_central.Write()
    h_eff_mc_altmc.Write()
    h_eff_mc_alttag.Write()
    h_eff_data.Write()
    h_eff_mc.Write()
    f.Close()


if __name__ == "__main__":
    main()
