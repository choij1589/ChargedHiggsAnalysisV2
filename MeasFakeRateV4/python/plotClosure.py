#!/home/choij/miniconda3/envs/pyg/bin/python
import os
import ROOT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="Skim1E2Mu / Skim3Mu")
parser.add_argument("--histkey", required=True, type=str, help="histkey, e.g. Central/ZCand/mass")
parser.add_argument("--rebin", required=False, type=int, default=5, help="rebin factor")
args = parser.parse_args()

## Settings
WORKDIR = os.environ['WORKDIR']
xtitle = ""
if args.histkey == "ZCand/mass":
    xtitle = "M(Z) [GeV]"
elif args.histkey == "nZCand/mass":
    xtitle = "M(nZ) [GeV]"
elif args.histkey == "electrons/1/pt":
    xtitle = "p_{T}(e) [GeV]"
elif args.histkey == "muons/1/pt":
    xtitle = "p_{T}(#mu1) [GeV]"
elif args.histkey == "muons/2/pt":
    xtitle = "p_{T}(#mu2) [GeV]"
elif args.histkey == "muons/3/pt":
    xtitle = "p_{T}(#mu3) [GeV]"
elif args.histkey == "electrons/1/scEta":
    xtitle = "#eta^{SC}(e)"
elif args.histkey == "muons/1/eta":
    xtitle = "#eta(#mu1)"
elif args.histkey == "muons/2/eta":
    xtitle = "#eta(#mu2)"
elif args.histkey == "muons/3/eta":
    xtitle = "#eta(#mu3)"
elif args.histkey == "nonprompt/pt":
    xtitle = "p_{T}(l^{fake}) [GeV]"
elif args.histkey == "nonprompt/eta":
    xtitle = "#eta(l^{fake})"
else:
    raise KeyError(f"Not registered histkey {args.histkey}")

## helper functions
def setInfoTo(text: ROOT.TLatex):
    text.SetTextSize(0.035)
    text.SetTextFont(42)

def setLogoTo(text: ROOT.TLatex):
    text.SetTextSize(0.04)
    text.SetTextFont(61)
    
def setWorkInProgressTo(text: ROOT.TLatex):
    text.SetTextSize(0.035)
    text.SetTextFont(52)
    
def setExtraInfoTo(text: ROOT.TLatex):
    text.SetTextSize(0.05)
    text.SetTextFont(42)

## Get histograms
f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/ClosFakeRate/{args.era}/{args.channel}__/ClosFakeRate_TTLL_powheg.root")
if args.channel == "Skim1E2Mu":
    h_obs = f.Get(f"SR1E2Mu/Central/{args.histkey}"); h_obs.SetDirectory(0)
    h_exp = f.Get(f"SB1E2Mu/Central/{args.histkey}"); h_exp.SetDirectory(0)
elif args.channel == "Skim3Mu":
    h_obs = f.Get(f"SR3Mu/Central/{args.histkey}"); h_obs.SetDirectory(0)
    h_exp = f.Get(f"SB3Mu/Central/{args.histkey}"); h_exp.SetDirectory(0)
else:
    raise KeyError(f"Wrong channel {args.channel}")
f.Close()

## Rebin
h_obs.Rebin(args.rebin)
h_exp.Rebin(args.rebin)

## Set error histogram
for bin in range(h_exp.GetNbinsX()):
    h_exp.SetBinError(bin+1, h_exp.GetBinContent(bin+1)*0.25)
h_err = h_exp.Clone("error")

h_obs.SetMarkerStyle(8)
h_obs.SetMarkerSize(2)
h_obs.SetMarkerColor(ROOT.kBlack)
h_exp.SetStats(0)
h_exp.SetLineColor(ROOT.kBlack)
h_exp.SetLineWidth(2)
h_err.SetStats(0)
h_err.SetFillColorAlpha(12, 0.9)
h_err.SetFillStyle(3144)
h_err.GetXaxis().SetTitle(xtitle)
if "[GeV]" in xtitle:
    h_err.GetYaxis().SetTitle(f"Events / {args.rebin} GeV")
else:
    h_err.GetYaxis().SetTitle("Events")

## Now draw comparison plots
canvas = ROOT.TCanvas("c", "", 1600, 1600)
canvas.SetLeftMargin(0.1)
canvas.SetRightMargin(0.08)
canvas.SetTopMargin(0.1)
canvas.SetBottomMargin(0.12)

legend = ROOT.TLegend(0.67, 0.65, 0.9, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize(0)

legend.AddEntry(h_obs, "observed", "plf")
legend.AddEntry(h_exp, "expected", "plf")
legend.AddEntry(h_err, "error, 25%", "f")

canvas.cd()
h_err.Draw("e2&f")
h_exp.Draw("hist&same")
h_obs.Draw("p&same")
legend.Draw("same")

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.805, 0.91, "(13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.1, 0.91, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.2, 0.91, "Work in progress")

outpath = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/plots/{args.channel}/closure_{args.histkey.replace('/', '_').lower()}.png"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
canvas.SaveAs(outpath)
