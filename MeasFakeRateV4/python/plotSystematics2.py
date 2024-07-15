#!/home/choij/miniconda3/envs/pyg/bin/python
import os
import argparse
import numpy as np
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--histkey", required=True, type=str, help="histogram key")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']
DATASTREAM = ""
if args.channel == "Skim1E2Mu":
    DATASTREAM = "MuonEG"
elif args.channel == "Skim3Mu":
    DATASTREAM = "DoubleMuon"
else:
    raise ValueError(f"Wrong channel {args.channel}")

LumiInfo = {    # /fb
        "2016preVFP": 19.5,
        "2016postVFP": 16.8,
        "2017": 41.5,
        "2018": 59.8
}

## helper functions
def getHisto(f: ROOT.TFile, histkey: str, syst: str, rebin: int=1) -> ROOT.TH1D:
    h = f.Get(f"{args.channel.replace('Skim', 'SR')}/{syst}/{histkey}")
    h.SetDirectory(0)
    h.SetStats(0)
    h.Rebin(rebin)
    return h
    
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
    
## Load histograms from MatrixEstimator
file_path = f"{WORKDIR}/SKFlatOutput/MatrixEstimator/{args.era}/{args.channel}__RunSyst__/DATA/MatrixEstimator_{DATASTREAM}.root" 
assert os.path.exists(file_path)
f = ROOT.TFile.Open(file_path)
h_Central = getHisto(f, args.histkey, "Central", 5)
h_PromptNormUp = getHisto(f, args.histkey, "PromptNormUp", 5)
h_PromptNormDown = getHisto(f, args.histkey, "PromptNormDown", 5)
h_MotherJetPtUp = getHisto(f, args.histkey, "MotherJetPtUp", 5)
h_MotherJetPtDown = getHisto(f, args.histkey, "MotherJetPtDown", 5)
h_RequireHeavyTag = getHisto(f, args.histkey, "RequireHeavyTag", 5)
f.Close()

## Settings for TAsymmGraph
nBins = h_Central.GetNbinsX()
systUp = np.zeros(nBins)
systDown = np.zeros(nBins)
x = np.zeros(nBins)
y = np.zeros(nBins)
ex = np.zeros(nBins)

for i in range(1, nBins+1):
    central = h_Central.GetBinContent(i)
    
    dPromptNormUp = h_PromptNormUp.GetBinContent(i) - central
    dPromptNormDown = h_PromptNormDown.GetBinContent(i) - central
    dMotherJetPtUp = h_MotherJetPtUp.GetBinContent(i) - central
    dMotherJetPtDown = h_MotherJetPtDown.GetBinContent(i) - central
    dRequireHeavyTag = h_RequireHeavyTag.GetBinContent(i) - central
    
    # Choose Up and Down variation
    dPromptNormPos = 0.
    dPromptNormNeg = 0.
    if dPromptNormUp > 0. and dPromptNormDown > 0.:
        dPromptNormPos = max(dPromptNormUp, dPromptNormDown)
        dPromptNormNeg = 0.
    elif dPromptNormUp > 0. and dPromptNormDown < 0.:
        dPromptNormPos, dPromptNormNeg = dPromptNormUp, dPromptNormDown
    elif dPromptNormUp < 0. and dPromptNormDown > 0.:
        dPromptNormPos, dPromptNormNeg = dPromptNormDown, dPromptNormUp
    else: # both negative
        dPromptNormPos = 0.
        dPromptNormNeg = min(dPromptNormUp, dPromptNormDown)
        
    dMotherJetPtPos = 0.
    dMotherJetPtNeg = 0.
    if dMotherJetPtUp > 0. and dMotherJetPtDown > 0.:
        dMotherJetPtPos = max(dMotherJetPtUp, dMotherJetPtDown)
        dMotherJetPtNeg = 0.
    elif dMotherJetPtUp > 0. and dMotherJetPtDown < 0.:
        dMotherJetPtPos, dMotherJetPtNeg = dMotherJetPtUp, dMotherJetPtDown
    elif dMotherJetPtDown < 0. and dMotherJetPtDown > 0.:
        dMotherJetPtPos, dMotherJetPtNeg = dMotherJetPtDown, dMotherJetPtUp
    else:
        dMotherJetPtPos = 0.
        dMotherJetPtNeg = min(dMotherJetPtUp, dMotherJetPtDown)
        
    systUp[i-1] = np.sqrt(
        dPromptNormPos**2 + dMotherJetPtPos**2 + (dRequireHeavyTag if dRequireHeavyTag > 0 else 0)**2 
    )
    systDown[i-1] = np.sqrt(
        dPromptNormNeg**2 + dMotherJetPtNeg**2 + (dRequireHeavyTag if dRequireHeavyTag < 0 else 0)**2 
    )
    
    # Prepare for TGraphAsymmErrors
    x[i-1] = h_Central.GetBinCenter(i)
    y[i-1] = central
    
# This plots are for estimating systematic errors
# So here we do not combine with stat errors
# It will be treated independently in combine tools
#totalUp = np.zeros(nBins)
#totalDown = np.zeros(nBins)
#for i in range(1, nBins+1):
#    stat = h_Central.GetBinError(i)
#    totalUp[i-1] = np.sqrt(stat**2 + systUp[i-1]**2)
#    totalDown[i-1] = np.sqrt(stat**2 + systDown[i-1]**2)

g = ROOT.TGraphAsymmErrors(nBins, x, y, ex, ex, systDown, systUp)
h_error = h_Central.Clone("error, 30%")
for i in range(1, h_error.GetNbinsX()+1):
    h_error.SetBinError(i, h_Central.GetBinContent(i)*0.3)
    
canvas = ROOT.TCanvas("c", "", 1600, 1600)
canvas.SetLeftMargin(0.1)
canvas.SetRightMargin(0.08)
canvas.SetTopMargin(0.1)
canvas.SetBottomMargin(0.12)

legend = ROOT.TLegend(0.67, 0.65, 0.9, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize(0)

h_Central.SetLineWidth(2)
h_Central.GetXaxis().SetTitle("M(#mu^{+}#mu^{-}) [GeV]")
if args.channel == "Skim1E2Mu":
    h_Central.GetYaxis().SetTitle("Events / 5 GeV")
elif args.channel == "Skim3Mu":
    h_Central.GetYaxis().SetTitle("Pairs / 5 GeV")
else:
    raise ValueError(f"Wrong channel {args.channel}")
h_Central.GetYaxis().SetTitleOffset(1.3)
h_Central.GetYaxis().SetRangeUser(0., h_Central.GetMaximum()*1.5)
h_error.SetMarkerStyle(20)
h_error.SetFillColorAlpha(ROOT.kGreen, 0.9)
h_error.SetFillStyle(3144)


g.SetMarkerStyle(20)
g.SetMarkerColor(ROOT.kBlack)
g.SetLineWidth(3)
g.SetLineColor(ROOT.kBlack)

legend.AddEntry(h_error, "error, 30%", "f")
legend.AddEntry(g, "error, syst", "pl")

canvas.cd()
h_Central.Draw("hist")
h_error.Draw("e2 fill same")
h_Central.Draw("hist same")
g.Draw("P same")
legend.Draw("same")

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.61, 0.91, "L_{int} ="+f" {LumiInfo[args.era]}"+" fb^{-1} (13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.1, 0.91, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.2, 0.91, "Work in progress")

outpath = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/plots/{args.channel}/systematics_{args.histkey.replace('/', '_')}.png"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
canvas.SaveAs(outpath)
