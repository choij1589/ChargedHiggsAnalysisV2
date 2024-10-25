import ROOT
from CombineHarvester.CombineTools.plotting import *
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(ROOT.kTRUE)

ERA = "2018"
CHANNEL = "Skim1E2Mu"
METHOD = "Shape"

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
