#!/usr/bin/env python3
import argparse
import ROOT
from CombineHarvester.CombineTools.plotting import *
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True, help="era")
parser.add_argument("--channel", type=str, required=True, help="channel")
parser.add_argument("--method", type=str, required=True, help="method")
parser.add_argument("--limit_type", type=str, required=True, help="Asymptotic / HybridNew")
args = parser.parse_args()

lumiDict = {
        "2016preVFP": "19.5",
        "2016postVFP": "16.8",
        "2017": "41.5",
        "2018": "59.8",
        "FullRun2": "138"
}

ModTDRStyle()
canvas = ROOT.TCanvas(f"limit.{args.era}.{args.channel}.{args.method}", "", 1000, 1000)
pads = OnePad()

# Get the limit TGraphs as a dictionary
graphs = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.json")
if args.method == "ParticleNet":
    graphs_pnet = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.ParticleNet.json")

axis = CreateAxisHist(list(graphs.values())[0])
axis.GetXaxis().SetTitle("m_{A} [GeV]")
axis.GetYaxis().SetTitle("95% CL limit on #sigma_{sig} [fb]")
pads[0].cd()
axis.Draw("axis")

# Create a legend in the top-left
legend = PositionedLegend(0.3, 0.2, 3, 0.015)

# Set the standard green and yellow colors
StyleLimitBand(graphs)
if args.method == "ParticleNet":
    DrawLimitBand(pads[0], graphs, draw=['exp2', 'exp1', 'exp0'])
else:
    DrawLimitBand(pads[0], graphs, draw=['exp2', 'exp1', 'exp0'], legend=legend)
graphs["exp0"].Draw("LSAME")

# For masspoint 85, 90, 95 are optimized with ParticleNet
# Re-draw the limits for ParticleNet
# In the edges (i.e. 85 and 95), draw both limits with discontinous line
if args.method == "ParticleNet":
    StyleLimitBand(graphs_pnet)
    DrawLimitBand(pads[0], graphs_pnet, draw=['exp2', 'exp1', 'exp0'], legend=legend)
    graphs_pnet["exp0"].Draw("LSAME")
    
    shape = graphs["exp0"]
    shape.SetLineColor(ROOT.kViolet)
    shape.SetLineWidth(2)
    shape.SetLineStyle(1)
    shape.Draw("LSAME")
    legend.AddEntry(shape, "Expected (Baseline)", "L")
    
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.DrawLine(85, 0, 85, GetPadYMax(pads[0]))
    line.DrawLine(95, 0, 95, GetPadYMax(pads[0]))

legend.Draw()
FixBothRanges(pads[0], 0, 0, GetPadYMax(pads[0]), 0.25)

DrawCMSLogo(pads[0], 'CMS', 'Internal', 11, 0.12, 0.035, 1.2, '', 0.8)
DrawTitle(pads[0], "L_{int} = "+lumiDict[args.era]+" fb^{-1}", 3)
# Re-draw the frame and tick marks
pads[0].RedrawAxis()
pads[0].GetFrame().Draw()

canvas.Print(".png")
