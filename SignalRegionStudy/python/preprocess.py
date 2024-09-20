#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--signal", required=True, type=str, help="Prepare for signal directory")
parser.add_argument("--debug", action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")

with open(f"{WORKDIR}/CommonData/json/convScaleFactors.json") as f:
    convScaleFactors = json.load(f)[args.era]
    
## helper functions
def hadd(file_path):
    if not os.path.exists(file_path):
        logging.info(os.listdir(f"{os.path.dirname(file_path)}"))
        logging.info("Hadding...")        
        os.system(f"hadd -f {file_path} {os.path.dirname(file_path)}/*")

bkgVV = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
bkgTTX = ["ttWToLNu", "ttZToLLNuNu", "ttHToNonbb"]
bkgConv = ["DYJets", "DYJets10to50_MG", "TTG", "WWG"]
bkgOthers = ["WWW", "WWZ", "WZZ", "ZZZ", "GluGluHToZZTo4L", "VBF_HToZZTo4L", "tZq", "tHq", "TTTT"]

promptSysts = [] 
matrixSysts = [] 
convSysts = [] 

if "1E2Mu" in args.channel:
    DATASTREAM = "MuonEG"
    promptSysts = [("Central",),
                   ("L1PrefireUp", "L1PrefireDown"),
                   ("PileupReweightUp", "PileupReweightDown"),
                   ("MuonIDSFUp", "MuonIDSFDown"),
                   ("ElectronIDSFUp", "ElectronIDSFDown"),
                   ("TriggerSFUp", "TriggerSFDown"),
                   ("ElectronResUp", "ElectronResDown"),
                   ("ElectronEnUp", "ElectronEnDown"),
                   ("JetResUp", "JetResDown"),
                   ("JetEnUp", "JetEnDown"),
                   ("MuonEnUp", "MuonEnDown")]
    convSF, convSFerr = convScaleFactors["1E2Mu"]
elif "3Mu" in args.channel:
    DATASTREAM = "DoubleMuon"
    promptSysts = [("Central",),
                   ("L1PrefireUp", "L1PrefireDown"),
                   ("PileupReweightUp", "PileupReweightDown"),
                   ("MuonIDSFUp", "MuonIDSFDown"),
                   ("TriggerSFUp", "TriggerSFDown"),
                   ("ElectronResUp", "ElectronResDown"),
                   ("ElectronEnUp", "ElectronEnDown"),
                   ("JetResUp", "JetResDown"),
                   ("JetEnUp", "JetEnDown"),
                   ("MuonEnUp", "MuonEnDown")]
    convSF, convSFerr = convScaleFactors["3Mu"]
else:
    raise ValueError(f"Invalid channel {args.channel}")

processor = ROOT.Preprocessor(args.era, args.channel, DATASTREAM)
processor.setConvSF(convSF, convSFerr)

## Signal
logging.info(f"Processing signal {args.signal}")
input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_TTToHcToWAToMuMu_{args.signal}.root"
output_path = f"samples/{args.era}/{args.channel}/{args.signal}/{args.signal}.root"
assert os.path.exists(input_path), f"Input file {input_path} does not exist"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

processor.setInputFile(input_path)
processor.setOutputFile(output_path)
for syst in [syst for systset in promptSysts for syst in systset]:
    processor.setInputTree(syst)
    processor.fillOutTree(args.signal, args.signal, syst, applyConvSF=False, isTrainedSample=False)
    processor.saveTree()
processor.closeInputFile()
processor.closeOutputFile()

## Nonprompt
logging.info("Processing nonprompt...")
input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MatrixSkimmer/{args.era}/{args.channel}__/DATA/MatrixSkimmer_{DATASTREAM}.root"
output_path = f"samples/{args.era}/{args.channel}/{args.signal}/nonprompt.root"
hadd(input_path); assert os.path.exists(input_path), f"Input file {input_path} does not exist"

processor.setInputFile(input_path)
processor.setOutputFile(output_path)
processor.setInputTree("Central")
processor.fillOutTree("nonprompt", args.signal, "Central", applyConvSF=False, isTrainedSample=False)
processor.saveTree()
processor.closeInputFile()
processor.closeOutputFile()

## Conversion
logging.info("Processing conversion...")
for sample in bkgConv:
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
    output_path = f"samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
    assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)
    processor.setInputTree("Central")
    processor.fillOutTree("conversion", args.signal, "Central", applyConvSF=True, isTrainedSample=False)
    processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
# hadd conversion samples to single file
hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/conversion.root"
for sample in bkgConv:
    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
os.system(hadd_line)

## diboson
logging.info("Processing diboson...")
for sample in bkgVV:
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
    output_path = f"samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
    assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)
    for syst in [syst for systset in promptSysts for syst in systset]:
        processor.setInputTree(syst)
        processor.fillOutTree("diboson", args.signal, syst, applyConvSF=False, isTrainedSample=False)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/diboson.root"
for sample in bkgVV:
    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
os.system(hadd_line)

## ttX
logging.info("Processing ttX...")
for sample in bkgTTX:
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
    output_path = f"samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
    assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)
    for syst in [syst for systset in promptSysts for syst in systset]:
        processor.setInputTree(syst)
        processor.fillOutTree("ttX", args.signal, syst, applyConvSF=False, isTrainedSample=False)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/ttX.root"
for sample in bkgTTX:
    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
os.system(hadd_line)

## others
logging.info("Processing others...")
for sample in bkgOthers:
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
    output_path = f"samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
    assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)
    for syst in [syst for systset in promptSysts for syst in systset]:
        processor.setInputTree(syst)
        processor.fillOutTree("others", args.signal, syst, applyConvSF=False, isTrainedSample=False)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/others.root"
for sample in bkgOthers:
    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
os.system(hadd_line)
