#!/usr/bin/env python
import os
import logging
import argparse
import numpy as np
import ROOT

WORKDIR = os.environ['WORKDIR']
parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--filter", required=True, type=str, help="DblMuDZ / DblMuDZM / DblMuM / EMuDZ")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug: logging.basicConfig(level=logging.DEBUG)

if args.filter == "DblMuDZ":
    if args.era == "2017":
        DATASTREAM = "DoubleMuon_B"
    else:
        DATASTREAM = "DoubleMuon"
    FLAG = "MeasDblMuDZ"
    DENOM = "TrigEff_Iso"
    NUM = "TrigEff_IsoDZ"
elif args.filter == "DblMuDZM":
    if args.era == "2017":
        DATASTREAM = "DoubleMuon_CDEF"
    else:
        DATASTREAM = "DoubleMuon"
    FLAG = "MeasDblMuDZ"
    DENOM = "TrigEff_Iso"
    NUM = "TrigEff_IsoM"
elif args.filter == "DblMuM":
    if args.era == "2017":
        DATASTREAM = "DoubleMuon_CDEF"
    else:
        DATASTREAM = "DoubleMuon"
    FLAG = "MeasDblMuDZ"
    DENOM = "TrigEff_IsoDZ"
    NUM = "TrigEff_IsoDZM"
elif args.filter == "EMuDZ":
    DATASTREAM = "MuonEG"
    FLAG = "MeasEMuDZ"
    # oring triggers, 1 for denominator, 2 for numerator
    DENOM = "TrigEff_EMuDZ"
    NUM = "TrigEff_EMuDZ"
else:
    raise ValueError(f"Invalid filter {args.filter}")

# helper functions
def meas_efficiency(filter: ROOT.TString, syst: ROOT.TString, is_data: bool) -> ROOT.TH1D:
    if is_data:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/DATA/MeasTrigEff_{DATASTREAM}.root")
    elif syst == "AltMC":
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/MeasTrigEff_DYJets.root")
    else:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/MeasTrigEff_TTLL_powheg.root")
    
    h_denom = f.Get(DENOM); h_denom.SetDirectory(0)
    h_num = f.Get(NUM); h_num.SetDirectory(0)
    f.Close()

    if filter == "EMuDZ":
        num, err_num = h_num.GetBinContent(2), h_num.GetBinError(2)
        denom, err_denom = h_denom.GetBinContent(1), h_denom.GetBinError(1)
    else:
        err_num = np.array([0.])
        err_denom = np.array([0.])
        num = h_num.IntegralAndError(1, h_num.GetNbinsX(), err_num)
        denom = h_denom.IntegralAndError(1, h_denom.GetNbinsX(), err_denom)
        err_num, err_denom = err_num[0], err_denom[0]
    eff = num / denom
    err = eff*np.sqrt((err_num/num)**2 + (err_denom/denom)**2)

    return (eff, err)
    
if __name__ == "__main__":
    print(f"# Pairwise Filter, Data Eff, MC Eff, MC Eff - AltMC")
    print(args.filter, meas_efficiency(args.filter, "Central", True), meas_efficiency(args.filter, "Central", False), meas_efficiency(args.filter, "AltMC", False), sep=", ")
