#!/usr/bin/env python
import os
import argparse
import ROOT
import tdrstyle; tdrstyle.setTDRStyle(square=False)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--isQCD", default=False, action="store_true", help="isQCD")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']
ptcorr_bins = []
abseta_bins = []
if args.measure == "muon":
    ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    abseta_bins = [0., 0.9, 1.6, 2.4]
elif args.measure == "electron":
    ptcorr_bins = [15., 20., 25., 35., 50., 100.]
    abseta_bins = [0., 0.8, 1.479, 2.5]
else:
    raise NameError(f"Wrong measure {args.measure}")

## Helper functions
def setInfoTo(text: ROOT.TLatex):
    text.SetTextSize(0.04)
    text.SetTextFont(42)

def setLogoTo(text: ROOT.TLatex):
    text.SetTextSize(0.05)
    text.SetTextFont(61)
    
def setWorkInProgressTo(text: ROOT.TLatex):
    text.SetTextSize(0.04)
    text.SetTextFont(52)
    
def setExtraInfoTo(text: ROOT.TLatex):
    text.SetTextSize(0.04)
    text.SetTextFont(42)
    
def setSystematics(fakerate: ROOT.TH1D):
    for bin in range(0, fakerate.GetNbinsX()+1):
        fakerate.SetBinError(bin, fakerate.GetBinContent(bin)*0.25)


## Get fakerate histogram
if args.isQCD:
    file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate_qcd.root"
else:
    file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate.root"

assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile(file_path)
if args.isQCD:
    h = f.Get("fake rate - (QCD_EMEnriched)") if args.measure == "electron" else f.Get("fake rate - (QCD_MuEnriched)")
else:
    h = f.Get("fake rate - (Central)")
h.SetDirectory(0)
f.Close()

## prepare canvas and legend
canvas = ROOT.TCanvas("c", "", 1200, 900)
#canvas.SetLeftMargin(0.1)
#canvas.SetRightMargin(0.08)
canvas.SetTopMargin(0.08)
canvas.SetBottomMargin(0.12)

legend = ROOT.TLegend(0.6, 0.55, 0.9, 0.8)
legend.SetFillStyle(0)
legend.SetBorderSize(0)

## prepare projections
projections = {}
projections["eta1"] = h.ProjectionY(f"eta{str(abseta_bins[0])}to{str(abseta_bins[1])}", 1, 1)
projections["eta2"] = h.ProjectionY(f"eta{str(abseta_bins[1])}to{str(abseta_bins[2])}", 2, 2)
projections["eta3"] = h.ProjectionY(f"eta{str(abseta_bins[2])}to{str(abseta_bins[3])}", 3, 3)

projections["eta1"].SetLineColor(ROOT.kRed)
projections["eta2"].SetLineColor(ROOT.kGreen)
projections["eta3"].SetLineColor(ROOT.kBlue)

setSystematics(projections["eta1"])
setSystematics(projections["eta2"])
setSystematics(projections["eta3"])

title = ""
if args.measure == "muon":     title = "fake rate (#mu)"
if args.measure == "electron": title = "fake rate (e)"
for hist in projections.values():
    hist.SetTitle("")
    hist.SetStats(0)
    hist.SetLineWidth(3)
    #hist.GetXaxis().SetLabelSize(0)
    hist.GetXaxis().SetTitle("p_{T}^{corr}")
    if args.measure == "muon":
        hist.GetXaxis().SetRangeUser(10., 50.)
    if args.measure == "electron":
        hist.GetXaxis().SetRangeUser(15., 50.)
    #if isQCD:
    #    hist.GetXaxis().SetRangeUser(10., 100.)
    hist.GetYaxis().SetRangeUser(0., 1.) 
    hist.GetYaxis().SetTitle(title)

legend.AddEntry(projections["eta1"], f"{abseta_bins[0]} < |#eta| < {abseta_bins[1]}", "lep")
legend.AddEntry(projections["eta2"], f"{abseta_bins[1]} < |#eta| < {abseta_bins[2]}", "lep")
legend.AddEntry(projections["eta3"], f"{abseta_bins[2]} < |#eta| < {abseta_bins[3]}", "lep")

ROOT.gStyle.SetErrorX(0.5)

canvas.cd()
projections["eta1"].Draw("")
projections["eta2"].Draw("same")
projections["eta3"].Draw("same")
canvas.RedrawAxis()
legend.Draw("same")

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.85, 0.93, "(13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.16, 0.925, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.25, 0.93, "Work in progress")
if args.isQCD:
    setExtraInfoTo(text); text.DrawLatexNDC(0.25, 0.83, "measured in QCD MC")

output_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/plots/{args.measure}/fakerate.png"
if args.isQCD:
    output_path = output_path.replace("fakerate", "fakerate_qcd")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)
