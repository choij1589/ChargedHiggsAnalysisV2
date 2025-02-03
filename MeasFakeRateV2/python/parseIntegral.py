#!/usr/bin/env python
import os
import argparse
import ROOT

# Parse command line arguments
parser = argparse.ArgumentParser(description='Parse QCD portion from fit result')
parser.add_argument('--era', type=str, required=True, help='Era')
parser.add_argument('--hlt', type=str, required=True, help='HLT')
parser.add_argument('--wp', type=str, required=True, help='WP')
parser.add_argument('--syst', type=str, required=True, help='Systematic')
args = parser.parse_args()

if args.hlt == "MeasFakeEl12":
    bins = ["ptcorr_15to20_EB1", "ptcorr_15to20_EB2", "ptcorr_15to20_EE",
            "ptcorr_20to25_EB1", "ptcorr_20to25_EB2", "ptcorr_20to25_EE",
            "ptcorr_25to35_EB1", "ptcorr_25to35_EB2", "ptcorr_25to35_EE",
            "ptcorr_35to50_EB1", "ptcorr_35to50_EB2", "ptcorr_35to50_EE",
            "ptcorr_50to100_EB1", "ptcorr_50to100_EB2", "ptcorr_50to100_EE",
            "ptcorr_100to200_EB1", "ptcorr_100to200_EB2", "ptcorr_100to200_EE"]
elif args.hlt == "MeasFakeEl23":
    bins = ["ptcorr_25to35_EB1", "ptcorr_25to35_EB2", "ptcorr_25to35_EE",
            "ptcorr_35to50_EB1", "ptcorr_35to50_EB2", "ptcorr_35to50_EE",
            "ptcorr_50to100_EB1", "ptcorr_50to100_EB2", "ptcorr_50to100_EE",
            "ptcorr_100to200_EB1", "ptcorr_100to200_EB2", "ptcorr_100to200_EE"]
elif "Mu" in args.hlt:
    pass
else:
    raise ValueError("HLT not recognized")

f_fitresult = ROOT.TFile.Open(f"output/{args.era}/{args.hlt}/fitresult.{args.wp}.{args.syst}.root")
f_data = ROOT.TFile.Open(f"../SKFlatOutput/MeasFakeRateV2/{args.era}/{args.hlt}__RunSystSimple__/DATA/MeasFakeRateV2_DoubleEG.root")
print("prefix, data, QCD_EMEnriched, QCD_bcToE")
for prefix in bins:
    fitresult = f_fitresult.Get(prefix);
    # fitresult.Print()
    init_pars = fitresult.floatParsInit()
    final_pars = fitresult.floatParsFinal()

    # find initial value of QCD portions
    QCD_EMEnriched = 0
    QCD_bcToE = 0

    init_check = 0
    final_check = 0
    for par in init_pars:
        init_check += par.getVal()
        if "QCD_EMEnriched" in par.GetName():
            QCD_EMEnriched = par.getVal()
        if "QCD_bcToE" in par.GetName():
            QCD_bcToE = par.getVal()

    for par in final_pars:
        final_check += par.getVal()
        if "QCD_EMEnriched" in par.GetName():
            QCD_EMEnriched *= par.getVal()
        if "QCD_bcToE" in par.GetName():
            QCD_bcToE *= par.getVal()
    h = f_data.Get(f"{prefix}/Inclusive/{args.wp}/{args.syst}/MT");
    data_integral = h.Integral()

    print(f"{prefix}, {data_integral:.6f}, {QCD_EMEnriched:.6f}, {QCD_bcToE:.6f}")
f_fitresult.Close()
f_data.Close()
