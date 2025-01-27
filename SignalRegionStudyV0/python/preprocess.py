#!/usr/bin/env python
import os, shutil
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

mA = int(args.signal.split("_")[1].split("-")[1])
isTrainedSample = (80 < mA and mA < 100)

## Number of variation for PDFs, Scale, and Parton Shower
NPDF = 100
NSCALE = 9
NPS = 4
    
## helper functions
def hadd(file_path):
    if not os.path.exists(file_path):
        logging.info(os.listdir(f"{os.path.dirname(file_path)}"))
        logging.info("Hadding...")        
        os.system(f"hadd -f {file_path} {os.path.dirname(file_path)}/*")

def getSampleAlias(sample):
    if sample == "WZTo3LNu_amcatnlo":
        return "WZ"
    elif sample == "ZZTo4L_powheg":
        return "ZZ"
    elif sample == "ttWToLNu":
        return "ttW"
    elif sample == "ttZToLLNuNu":
        return "ttZ"
    elif sample == "ttHToNonbb":
        return "ttH"
    else:
        return sample

bkgVV = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
bkgTTX = ["ttWToLNu", "ttZToLLNuNu", "ttHToNonbb", "tZq"]
bkgConv = ["DYJets", "DYJets10to50_MG", "TTG", "WWG"]
bkgOthers = ["WWW", "WWZ", "WZZ", "ZZZ", "GluGluHToZZTo4L", "VBF_HToZZTo4L", "tHq", "TTTT"]

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
theorySysts = [("AlpS_up", "AlpS_down"),
                ("AlpSfact_up", "AlpSfact_down"),
               tuple([f"PDFReweight_{i}" for i in range(NPDF)]),
               tuple([f"ScaleVar_{i}" for i in [0, 1, 2, 3, 4, 6, 8]]),
               tuple([f"PSVar_{i}" for i in range(NPS)])
               ]

processor = ROOT.Preprocessor(args.era, args.channel, DATASTREAM)
processor.setConvSF(convSF, convSFerr)

## Signal
logging.info(f"Processing signal {args.signal}")
input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__RunTheoryUnc__/PromptSkimmer_TTToHcToWAToMuMu_{args.signal}.root"
output_path = f"samples/{args.era}/{args.channel}/{args.signal}/{args.signal}.root"
assert os.path.exists(input_path), f"Input file {input_path} does not exist"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

processor.setInputFile(input_path)
processor.setOutputFile(output_path)
for syst in [syst for systset in promptSysts for syst in systset]:
    processor.setInputTree(syst)
    processor.fillOutTree(args.signal, args.signal, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
    processor.saveTree()
for syst in [syst for systset in theorySysts for syst in systset]:
    processor.setInputTree(syst)
    processor.fillOutTree(args.signal, args.signal, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
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
processor.fillOutTree("nonprompt", args.signal, "Central", applyConvSF=False, isTrainedSample=isTrainedSample)
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
    processor.fillOutTree("conversion", args.signal, "Central", applyConvSF=True, isTrainedSample=isTrainedSample)
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
        processor.fillOutTree(getSampleAlias(sample), args.signal, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
### We will treat cross-section normalization separately for diboson, just rename the samples
shutil.move(f"samples/{args.era}/{args.channel}/{args.signal}/WZTo3LNu_amcatnlo.root", f"samples/{args.era}/{args.channel}/{args.signal}/WZ.root")
shutil.move(f"samples/{args.era}/{args.channel}/{args.signal}/ZZTo4L_powheg.root", f"samples/{args.era}/{args.channel}/{args.signal}/ZZ.root")

#hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/diboson.root"
#for sample in bkgVV:
#    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
#os.system(hadd_line)

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
        processor.fillOutTree(getSampleAlias(sample), args.signal, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
### We will treat cross-section normalization separately for diboson, just rename the samples
shutil.move(f"samples/{args.era}/{args.channel}/{args.signal}/ttWToLNu.root", f"samples/{args.era}/{args.channel}/{args.signal}/ttW.root")
shutil.move(f"samples/{args.era}/{args.channel}/{args.signal}/ttZToLLNuNu.root", f"samples/{args.era}/{args.channel}/{args.signal}/ttZ.root")
shutil.move(f"samples/{args.era}/{args.channel}/{args.signal}/ttHToNonbb.root", f"samples/{args.era}/{args.channel}/{args.signal}/ttH.root")
shutil.move(f"samples/{args.era}/{args.channel}/{args.signal}/tZq.root", f"samples/{args.era}/{args.channel}/{args.signal}/tZq.root")

#hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/ttX.root"
#for sample in bkgTTX:
#    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
#os.system(hadd_line)

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
        processor.fillOutTree("others", args.signal, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()
hadd_line = f"hadd -f samples/{args.era}/{args.channel}/{args.signal}/others.root"
for sample in bkgOthers:
    hadd_line += f" samples/{args.era}/{args.channel}/{args.signal}/{sample}.root"
os.system(hadd_line)
