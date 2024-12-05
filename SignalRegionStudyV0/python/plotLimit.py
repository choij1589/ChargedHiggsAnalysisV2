#!/usr/bin/env python3
import ROOT
from CombineHarvester.CombineTools.plotting import *
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

ERA = "2016preVFP"
CHANNEL = "SR1E2Mu"
METHOD = "Baseline"

lumiDict = {
        "2016preVFP": "19.5",
        "2016postVFP": "16.8",
        "2017": "41.5",
        "2018": "59.8",
        "FullRun2": "138"
}

ModTDRStyle()
canvas = ROOT.TCanvas(f"limit.{ERA}.{CHANNEL}.{METHOD}")
pads = OnePad()

# Get the limit TGraphs as a dictionary
graphs = StandardLimitsFromJSONFile(f"results/json/limits.{ERA}.{CHANNEL}.Asymptotic.Baseline.json")
#graphs_gnn = StandardLimitsFromJSONFile(f"results/json/limits.{ERA}.{CHANNEL}.Asymptotic.ParticleNet.json")

axis = CreateAxisHist(list(graphs.values())[0])
axis.GetXaxis().SetTitle("m_{A} [GeV]")
axis.GetYaxis().SetTitle("95% CL limit on #sigma_{sig} [fb]")
pads[0].cd()
axis.Draw("axis")

# Create a legend in the top-left
legend = PositionedLegend(0.3, 0.2, 3, 0.015)

# Set the standard green and yellow colors
StyleLimitBand(graphs)
DrawLimitBand(pads[0], graphs, legend=legend)
legend.Draw()

FixBothRanges(pads[0], 0, 0, GetPadYMax(pads[0]), 0.25)

DrawCMSLogo(pads[0], 'CMS', 'Internal', 11, 0.045, 0.035, 1.2, '', 0.8)

canvas.Print(".pdf")
canvas.Print(".png")
