#!/usr/bin/env python3
import os
import argparse
import ROOT
import json
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

MASSPOINTs_BELOWZ = [15, 40, 55, 60, 65, 85]
MASSPOINTs_ONZ = [85, 90, 95]
MASSPOINTs_ABOVEZ = [95, 120, 125, 155]

def split_json():
    limits_belowZ = {}
    limits_onZ = {}
    limits_aboveZ_baseline = {}
    # parse results from json
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.json") as f:
        limits = json.load(f)
        for mA in MASSPOINTs_BELOWZ:
            limits_belowZ[mA] = limits[str(mA)]
        for mA in MASSPOINTs_ONZ:
            limits_onZ[mA] = limits[str(mA)]
        for mA in MASSPOINTs_ABOVEZ:
            limits_aboveZ_baseline[mA] = limits[str(mA)]
    # write to json
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.BelowZ.json", "w") as f:
        json.dump(limits_belowZ, f, indent=2)
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.OnZ.json", "w") as f:
        json.dump(limits_onZ, f, indent=2)
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.AboveZ.json", "w") as f:
        json.dump(limits_aboveZ_baseline, f, indent=2)
    
ModTDRStyle()
canvas = ROOT.TCanvas(f"limit.{args.era}.{args.channel}.{args.method}", "", 1000, 1000)
pads = OnePad()

if args.method == "Baseline":
    graphs = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.json")
elif args.method == "ParticleNet":
    split_json()
    graphs = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.json")
    graphs_belowZ = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.BelowZ.json")
    graphs_onZ = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.OnZ.json")
    graphs_aboveZ = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.AboveZ.json")
    graphs_pnet = StandardLimitsFromJSONFile(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.ParticleNet.json")
else:
    raise ValueError(f"Method {args.method} is not supported")

axis = CreateAxisHist(list(graphs.values())[0])
axis.GetXaxis().SetTitle("m_{A} [GeV]")
axis.GetYaxis().SetTitle("95% CL limit on #sigma_{sig} [fb]")
pads[0].cd()
axis.Draw("axis")

# Create a legend in the top-left
legend = PositionedLegend(0.3, 0.2, 3, 0.015)

# Set the standard green and yellow colors

if args.method == "Baseline":
    StyleLimitBand(graphs)
    DrawLimitBand(pads[0], graphs, draw=['exp2', 'exp1', 'exp0'], legend=legend)
elif args.method == "ParticleNet":
    StyleLimitBand(graphs_pnet)
    DrawLimitBand(pads[0], graphs_pnet, draw=['exp2', 'exp1', 'exp0'], legend=legend)
    StyleLimitBand(graphs_belowZ)
    DrawLimitBand(pads[0], graphs_belowZ, draw=['exp2', 'exp1', 'exp0'])
    StyleLimitBand(graphs_aboveZ)
    DrawLimitBand(pads[0], graphs_aboveZ, draw=['exp2', 'exp1', 'exp0'])
    shape = graphs_onZ["exp0"]
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
else:
    raise ValueError(f"Method {args.method} is not supported")
    
legend.Draw()
FixBothRanges(pads[0], 0, 0, GetPadYMax(pads[0]), 0.25)

DrawCMSLogo(pads[0], 'CMS', 'Internal', 11, 0.12, 0.035, 1.2, '', 0.8)
DrawTitle(pads[0], "L_{int} = "+lumiDict[args.era]+" fb^{-1}", 3)
# Re-draw the frame and tick marks
pads[0].RedrawAxis()
pads[0].GetFrame().Draw()

output_name = f"results/plots/limit.{args.era}.{args.channel}.{args.method}.{args.limit_type}.png"
os.makedirs(os.path.dirname(output_name), exist_ok=True)
canvas.SaveAs(output_name)