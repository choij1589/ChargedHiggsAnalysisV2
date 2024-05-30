import os
import argparse
import ROOT
ROOT.gSystem.Load("lib/libMeasFakeRate.dylib")

parser = argparse.ArgumentParser(description='Perform fit')
parser.add_argument('--era', type=str, required=True, help='Era')
parser.add_argument('--hlt', type=str, required=True, help='HLT path')
parser.add_argument('--wp', type=str, required=True, help='Working point')
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

fitter = ROOT.FitMT(args.era, args.hlt, args.wp, args.syst)
outname = f"output/{args.era}/{args.hlt}/fitresult.{args.wp}.{args.syst}.root"
os.makedirs(os.path.dirname(outname), exist_ok=True)
f = ROOT.TFile(outname, "RECREATE")
for prefix in bins:
    res = fitter.fit(prefix)
    f.cd()
    res.Write(prefix)
f.Close()
