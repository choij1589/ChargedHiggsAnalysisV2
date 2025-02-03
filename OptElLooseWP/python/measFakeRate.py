#!/usr/bin/env python
import os
import argparse
import ROOT
from array import array
from DataFormat import Electron

## Note - only electron needs re-optimization
parser = argparse.ArgumentParser(description="Measure fake rate based on TTbar MC")
parser.add_argument("--era", type=str, required=True, help="Era")
parser.add_argument("--region", type=str, required=True, help="eta region")
parser.add_argument("--is_old", action="store_true", default=False, help="Use old MVA cut")
args = parser.parse_args()

WORKDIR = os.environ["WORKDIR"]

## Helper functions
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
    text.SetTextSize(0.035)
    text.SetTextFont(42)

# Define binning
ptbins = [15., 20., 25., 35., 50., 100.]
h_ljet_loose = ROOT.TH1F("ljet_loose", "", len(ptbins)-1, array('d', ptbins)); h_ljet_loose.SetDirectory(0)
h_cjet_loose = ROOT.TH1F("cjet_loose", "", len(ptbins)-1, array('d', ptbins)); h_cjet_loose.SetDirectory(0)
h_bjet_loose = ROOT.TH1F("bjet_loose", "", len(ptbins)-1, array('d', ptbins)); h_bjet_loose.SetDirectory(0)

h_ljet_tight = ROOT.TH1F("ljet_tight", "", len(ptbins)-1, array('d', ptbins)); h_ljet_tight.SetDirectory(0)
h_cjet_tight = ROOT.TH1F("cjet_tight", "", len(ptbins)-1, array('d', ptbins)); h_cjet_tight.SetDirectory(0)
h_bjet_tight = ROOT.TH1F("bjet_tight", "", len(ptbins)-1, array('d', ptbins)); h_bjet_tight.SetDirectory(0)

f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/OptElLooseWP/{args.era}/OptElLooseWP_TTLL_powheg.root")
for evt in f.Events:
    electrons = []
    for i in range(evt.nElectrons):
        el = Electron(args.era, args.region, args.is_old)
        el.setPt(evt.Pt[i])
        el.setScEta(evt.scEta[i])
        el.setMVANoIso(evt.MVANoIso[i])
        el.setMiniRelIso(evt.MiniRelIso[i])
        el.setSIP3D(evt.SIP3D[i])
        el.setDeltaR(evt.DeltaR[i])
        el.setID(evt.PassMVANoIsoWP90[i], evt.PassMVANoIsoWPLoose[i])
        el.setNearestJetFlavour(evt.NearestJetFlavour[i])
        el.setPtCorr()
        el.setMVACut()
        if not el.is_valid_region(): continue
        electrons.append(el)
        
    # classify electrons depending on mother jet flavour
    for el in electrons:
        if el.deltaR > 0.4: continue
        if not el.passLooseID(): continue
        
        if el.nearestJetFlavour == 1:
            h_ljet_loose.Fill(el.ptCorr, evt.genWeight)
            if el.passTightID(): 
                h_ljet_tight.Fill(el.ptCorr, evt.genWeight)
        elif el.nearestJetFlavour == 4:
            h_cjet_loose.Fill(el.ptCorr, evt.genWeight)
            if el.passTightID(): 
                h_cjet_tight.Fill(el.ptCorr, evt.genWeight)
        elif el.nearestJetFlavour == 5:
            h_bjet_loose.Fill(el.ptCorr, evt.genWeight)
            if el.passTightID(): 
                h_bjet_tight.Fill(el.ptCorr, evt.genWeight)
        else:
            continue
f.Close()       
        

## Calculate fake rate
fake_ljet = h_ljet_tight.Clone("fake_ljet"); fake_ljet.Divide(h_ljet_loose)
fake_cjet = h_cjet_tight.Clone("fake_cjet"); fake_cjet.Divide(h_cjet_loose)
fake_bjet = h_bjet_tight.Clone("fake_bjet"); fake_bjet.Divide(h_bjet_loose)

fake_ljet.SetStats(0)
fake_ljet.SetLineColor(ROOT.kBlack)
fake_ljet.SetLineWidth(3)
fake_ljet.GetXaxis().SetRangeUser(15., 100.)
fake_ljet.GetYaxis().SetRangeUser(0., 1.)

fake_cjet.SetStats(0)
fake_cjet.SetLineColor(ROOT.kGreen)
fake_cjet.SetLineWidth(3)

fake_bjet.SetStats(0)
fake_bjet.SetLineColor(ROOT.kBlue)
fake_bjet.SetLineWidth(3)

# Set 30% systematics in light fake rate
for bin in range(0, fake_ljet.GetNbinsX()+1):
    fake_ljet.SetBinError(bin, fake_ljet.GetBinContent(bin)*0.3)
    
canvas = ROOT.TCanvas("c", "", 1600, 1200)
canvas.SetLeftMargin(0.1)
canvas.SetRightMargin(0.08)
canvas.SetTopMargin(0.1)
canvas.SetBottomMargin(0.12)

legend = ROOT.TLegend(0.67, 0.65, 0.9, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize(0)

title = "fake rate (e)"
fake_ljet.GetXaxis().SetTitle("p_{T}^{corr}")
fake_ljet.GetYaxis().SetTitle(title)

legend.AddEntry(fake_ljet, "l-jet, 30% syst.", "lep")
legend.AddEntry(fake_cjet, "c-jet", "lp")
legend.AddEntry(fake_bjet, "b-jet", "lp")
    
    
canvas.cd()
fake_ljet.Draw()
fake_cjet.Draw("hist&same")
fake_bjet.Draw("hist&same")
legend.Draw("same")

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.835, 0.91, "(13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.1, 0.91, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.18, 0.91, "Work in progress")
setExtraInfoTo(text); text.DrawLatexNDC(0.15, 0.83, "measured in TTbar MC")

out_path = f"{WORKDIR}/OptElLooseWP/plots/{args.era}/fakeRate_{args.region}_newMVA.png"
if args.is_old:
    out_path = out_path.replace("_newMVA.png", "_oldMVA.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
canvas.SaveAs(out_path)
