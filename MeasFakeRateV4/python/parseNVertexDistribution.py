#!/home/choij/miniconda3/envs/pyg/bin/python
import os
import argparse
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="Era")
parser.add_argument("--hlt", required=True, type=str, help="HLT path")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']
DATASTREAM = ""
if "El" in args.hlt:
    if args.era in ["2016preVFP", "2016postVFP"]:
        DATASTREAM = "DoubleEG"
    elif args.era == "2017":
        DATASTREAM = "SingleElectron"
    elif args.era == "2018":
        DATASTREAM = "EGamma"
    else:
        raise ValueError("Invalid era")
    out_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/electron/NPV_{args.hlt}.root"
elif "Mu" in args.hlt:
    out_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/muon/NPV_{args.hlt}.root"
    DATASTREAM = "DoubleMuon"

# get data and MC files
histkey = "Inclusive/loose/Central/nPV"
data = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{args.hlt}__/DATA/MeasFakeRateV4_{DATASTREAM}.root")
mc = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/MeasFakeRateV4/{args.era}/{args.hlt}__/MeasFakeRateV4_WJets_MG.root")

h_data = data.Get(histkey); h_data.SetDirectory(0); data.Close();
h_data.SetTitle("NPV_Data")
h_mc = mc.Get(histkey); h_mc.SetDirectory(0); mc.Close();
h_mc.SetTitle("NPV_MC")

h_data.Scale(1./h_data.Integral())
h_mc.Scale(1./h_mc.Integral())

ratio = h_data.Clone("NPV_SF")
ratio.Divide(h_mc)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
out = ROOT.TFile.Open(out_path, "RECREATE")
h_data.Write("NPV_Data")
h_mc.Write("NPV_MC")
ratio.Write("NPV_SF")
out.Close()

