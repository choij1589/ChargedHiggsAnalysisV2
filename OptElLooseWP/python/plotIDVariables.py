#!/home/choij/miniconda3/envs/pyg/bin/python
import os
import argparse
import ROOT
from DataFormat import Electron

WORKDIR = os.environ["WORKDIR"]
parser = argparse.ArgumentParser(description='Plot ID variables')
parser.add_argument('--era', type=str, required=True, help='Era')
parser.add_argument('--region', type=str, required=True, help='eta region')
args = parser.parse_args()

## Helper function
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
    
def setStyle(h_ljet, h_cjet, h_bjet):
    h_ljet.SetStats(0); h_ljet.Scale(1./h_ljet.Integral())
    h_cjet.SetStats(0); h_cjet.Scale(1./h_cjet.Integral())
    h_bjet.SetStats(0); h_bjet.Scale(1./h_bjet.Integral())
    
    h_ljet.SetLineColor(ROOT.kBlack); h_ljet.SetLineWidth(3)
    h_cjet.SetLineColor(ROOT.kGreen); h_cjet.SetLineWidth(3)
    h_bjet.SetLineColor(ROOT.kBlue); h_bjet.SetLineWidth(3)
    
    h_ljet.GetYaxis().SetRangeUser(1e-4, 1.)

## Register histograms
h_ljet_mIso = ROOT.TH1F("miniRelIso_ljet", "", 70, 0, 0.7); h_ljet_mIso.SetDirectory(0)
h_cjet_mIso = ROOT.TH1F("miniRelIso_cjet", "", 70, 0, 0.7); h_cjet_mIso.SetDirectory(0)
h_bjet_mIso = ROOT.TH1F("miniRelIso_bjet", "", 70, 0, 0.7); h_bjet_mIso.SetDirectory(0)

# mvaNoIso
h_ljet_mva = ROOT.TH1F("mvaNoIso_ljet", "", 200, -1, 1); h_ljet_mva.SetDirectory(0)
h_cjet_mva = ROOT.TH1F("mvaNoIso_cjet", "", 200, -1, 1); h_cjet_mva.SetDirectory(0)
h_bjet_mva = ROOT.TH1F("mvaNoIso_bjet", "", 200, -1, 1); h_bjet_mva.SetDirectory(0)

# SIP3D
h_ljet_sip3d = ROOT.TH1F("mvaNoIso_ljet", "", 100, 0, 10); h_ljet_sip3d.SetDirectory(0)
h_cjet_sip3d = ROOT.TH1F("mvaNoIso_cjet", "", 100, 0, 10); h_cjet_sip3d.SetDirectory(0)
h_bjet_sip3d = ROOT.TH1F("mvaNoIso_bjet", "", 100, 0, 10); h_bjet_sip3d.SetDirectory(0)

## Fill histograms
f = ROOT.TFile.Open(f"{WORKDIR}/SKFlatOutput/OptElLooseWP/{args.era}/OptElLooseWP_TTLL_powheg.root")
for evt in f.Events:
    electrons = []
    for i in range(evt.nElectrons):
        el = Electron(args.era, args.region, is_old=True)
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
    
    for el in electrons:
        if not el.deltaR < 0.4: continue
        if el.nearestJetFlavour == 5:
            h_bjet_mIso.Fill(el.miniRelIso, evt.genWeight)
            h_bjet_mva.Fill(el.mvaNoIso, evt.genWeight)
            h_bjet_sip3d.Fill(el.sip3d, evt.genWeight)
        elif el.nearestJetFlavour == 4:
            h_cjet_mIso.Fill(el.miniRelIso, evt.genWeight)
            h_cjet_mva.Fill(el.mvaNoIso, evt.genWeight)
            h_cjet_sip3d.Fill(el.sip3d, evt.genWeight)
        elif el.nearestJetFlavour == 1:
            h_ljet_mIso.Fill(el.miniRelIso, evt.genWeight)
            h_ljet_mva.Fill(el.mvaNoIso, evt.genWeight)
            h_ljet_sip3d.Fill(el.sip3d, evt.genWeight)
        else:
            continue
f.Close()

## Draw one by one
canvas = ROOT.TCanvas("canvas", "", 1300, 1400)
legend = ROOT.TLegend(0.67, 0.65, 0.9, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize()

setStyle(h_ljet_mIso, h_cjet_mIso, h_bjet_mIso)
h_ljet_mIso.GetXaxis().SetTitle("Iso_{mini}^{rel}")
h_ljet_mIso.GetYaxis().SetTitle("A.U.")
h_ljet_mIso.GetYaxis().SetTitleOffset(0.8)
h_ljet_mIso.GetXaxis().SetRangeUser(0., 0.6)
h_ljet_mIso.GetXaxis().SetTitleOffset(1.2)
legend.AddEntry(h_ljet_mIso, "l-jet", "l")
legend.AddEntry(h_cjet_mIso, "c-jet", "l")
legend.AddEntry(h_bjet_mIso, "b-jet", "l")

canvas.cd()
canvas.SetLogy()
h_ljet_mIso.Draw("hist")
h_cjet_mIso.Draw("hist same")
h_bjet_mIso.Draw("hist same")
h_ljet_mIso.Draw("hist same")
legend.Draw("same")
canvas.RedrawAxis()

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.835, 0.91, "(13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.1, 0.91, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.18, 0.91, "Work in progress")
setExtraInfoTo(text); text.DrawLatexNDC(0.15, 0.83, "measured in TTbar MC")

out_path = f"{WORKDIR}/OptElLooseWP/plots/{args.era}/miniIso_{args.region}.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
canvas.SaveAs(out_path)

## SIP3D
canvas = ROOT.TCanvas("canvas", "", 1300, 1400)
legend = ROOT.TLegend(0.67, 0.65, 0.9, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize()

setStyle(h_ljet_sip3d, h_cjet_sip3d, h_bjet_sip3d)
h_ljet_sip3d.GetXaxis().SetTitle("SIP3D")
h_ljet_sip3d.GetYaxis().SetTitle("A.U.")
h_ljet_sip3d.GetYaxis().SetTitleOffset(0.8)
h_ljet_sip3d.GetXaxis().SetTitleOffset(1.2)
h_ljet_sip3d.GetXaxis().SetRangeUser(0., 8.)
legend.AddEntry(h_ljet_sip3d, "l-jet", "l")
legend.AddEntry(h_cjet_sip3d, "c-jet", "l")
legend.AddEntry(h_bjet_sip3d, "b-jet", "l")

canvas.cd()
canvas.SetLogy()
h_ljet_sip3d.Draw("hist")
h_cjet_sip3d.Draw("hist same")
h_bjet_sip3d.Draw("hist same")
h_ljet_sip3d.Draw("hist same")
legend.Draw("same")
canvas.RedrawAxis()

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.835, 0.91, "(13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.1, 0.91, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.18, 0.91, "Work in progress")
setExtraInfoTo(text); text.DrawLatexNDC(0.15, 0.83, "measured in TTbar MC")

out_path = f"{WORKDIR}/OptElLooseWP/plots/{args.era}/SIP3D_{args.region}.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
canvas.SaveAs(out_path)

## MAVNoIso
## SIP3D
canvas = ROOT.TCanvas("canvas", "", 1300, 1400)
legend = ROOT.TLegend(0.67, 0.65, 0.9, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize()

#h_ljet_mva.Rebin(2)
#h_cjet_mva.Rebin(2)
#h_bjet_mva.Rebin(2)
setStyle(h_ljet_mva, h_cjet_mva, h_bjet_mva)
h_ljet_mva.GetXaxis().SetTitle("mvaNoIso")
h_ljet_mva.GetYaxis().SetTitle("A.U.")
h_ljet_mva.GetYaxis().SetTitleOffset(0.8)
h_ljet_mva.GetXaxis().SetTitleOffset(1.2)
h_ljet_mva.GetXaxis().SetRangeUser(-0.8, 1.)
legend.AddEntry(h_ljet_mva, "l-jet", "l")
legend.AddEntry(h_cjet_mva, "c-jet", "l")
legend.AddEntry(h_bjet_mva, "b-jet", "l")

canvas.cd()
canvas.SetLogy()
h_ljet_mva.Draw("hist")
h_cjet_mva.Draw("hist same")
h_bjet_mva.Draw("hist same")
h_ljet_mva.Draw("hist same")
legend.Draw("same")
canvas.RedrawAxis()

text = ROOT.TLatex()
setInfoTo(text); text.DrawLatexNDC(0.835, 0.91, "(13TeV)")
setLogoTo(text); text.DrawLatexNDC(0.1, 0.91, "CMS")
setWorkInProgressTo(text); text.DrawLatexNDC(0.18, 0.91, "Work in progress")
setExtraInfoTo(text); text.DrawLatexNDC(0.15, 0.83, "measured in TTbar MC")

out_path = f"{WORKDIR}/OptElLooseWP/plots/{args.era}/mvaNoIso_{args.region}.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
canvas.SaveAs(out_path)


