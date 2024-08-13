#!/usr/bin/env python
import os
import logging
import argparse
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
    FLAG = "MeasDblMuM"
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
def get_histograms(filter: ROOT.TString, syst: ROOT.TString, is_data: bool) -> ROOT.TH1D:
    if is_data:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/DATA/MeasTrigEff_{DATASTREAM}.root")
    elif syst == "AltMC":
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/MeasTrigEff_DYJets.root")
    else:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MeasTrigEff/{args.era}/{FLAG}__/MeasTrigEff_TTLL_powheg.root")
    
    h_denom = f.Get(DENOM); h_denom.SetDirectory(0)
    h_num = f.Get(NUM); h_num.SetDirectory(0)
    f.Close()
    
    return (h_denom, h_num)

if __name__ == "__main__":
    h_denom, h_num = get_histograms(args.filter, "Central", True)