#!/usr/bin/env python
import os, shutil
import argparse
import logging
import json
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--masspoint", required=True, type=str, help="signal mass point")
parser.add_argument("--method", required=True, type=str, help="method")
parser.add_argument("--debug", action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")
BASEDIR = f"{WORKDIR}/SignalRegionStudyV1/samples/{args.era}/{args.channel}/{args.masspoint}/{args.method}"

with open(f"{WORKDIR}/CommonData/json/convScaleFactors.json") as f:
    convScaleFactors = json.load(f)[args.era]
    
mA = int(args.masspoint.split("_")[1].split("-")[1])
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
    sample_aliases = {
        "WZTo3LNu_amcatnlo": "WZ",
        "ZZTo4L_powheg": "ZZ",
        "ttWToLNu": "ttW",
        "ttZToLLNuNu": "ttZ",
        "ttHToNonbb": "ttH"
    }
    return sample_aliases.get(sample, sample)


DIBOSONBKGs = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
TTXBKGs = ["ttWToLNu", "ttZToLLNuNu", "ttHToNonbb", "tZq"]
CONVBKGs = ["DYJets", "DYJets10to50_MG", "TTG", "WWG"]
RAREBKGs = ["WWW", "WWZ", "WZZ", "ZZZ", "GluGluHToZZTo4L", "VBF_HToZZTo4L", "tHq", "TTTT"]

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

if __name__ == "__main__":
    logging.info(f"Preprocessing signal {args.masspoint} for {args.era} era and {args.channel} channel")
    if os.path.exists(BASEDIR):
        logging.info(f"Removing existing directory {BASEDIR}")
        shutil.rmtree(BASEDIR)
    os.makedirs(BASEDIR, exist_ok=True)

    processor = ROOT.Preprocessor(args.era, args.channel, DATASTREAM)
    processor.setConvSF(convSF, convSFerr)

    ## Signal
    logging.info(f"Processing signal {args.masspoint}")
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__RunTheoryUnc__/PromptSkimmer_TTToHcToWAToMuMu_{args.masspoint}.root"
    output_path = f"{BASEDIR}/{args.masspoint}.root"
    assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)
    for syst in [syst for systset in promptSysts for syst in systset]:
        processor.setInputTree(syst)
        processor.fillOutTree(args.masspoint, args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()
    for syst in [syst for systset in theorySysts for syst in systset]:
        processor.setInputTree(syst)
        processor.fillOutTree(args.masspoint, args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()

    ## Nonprompt
    logging.info("Processing nonprompt...")
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/MatrixSkimmer/{args.era}/{args.channel}__/DATA/MatrixSkimmer_{DATASTREAM}.root"
    output_path = f"{BASEDIR}/nonprompt.root"
    hadd(input_path); assert os.path.exists(input_path), f"Input file {input_path} does not exist"

    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)
    processor.setInputTree("Central")
    processor.fillOutTree("nonprompt", args.masspoint, "Central", applyConvSF=False, isTrainedSample=isTrainedSample)
    processor.saveTree()
    processor.closeInputFile()
    processor.closeOutputFile()

    ## Conversion
    logging.info("Processing conversion...")
    for sample in CONVBKGs:
        input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
        output_path = f"{BASEDIR}/{sample}.root"
        assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)
        processor.setInputTree("Central")
        processor.fillOutTree("conversion", args.masspoint, "Central", applyConvSF=True, isTrainedSample=isTrainedSample)
        processor.saveTree()
        processor.closeInputFile()
        processor.closeOutputFile()
    # hadd conversion samples to single file
    hadd_line = f"hadd -f {BASEDIR}/conversion.root"
    for sample in CONVBKGs:
        hadd_line += f" {BASEDIR}/{sample}.root"
    os.system(hadd_line)

    ## diboson
    logging.info("Processing diboson...")
    for sample in DIBOSONBKGs:
        input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
        output_path = f"{BASEDIR}/{sample}.root"
        assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)
        for syst in [syst for systset in promptSysts for syst in systset]:
            processor.setInputTree(syst)
            processor.fillOutTree(getSampleAlias(sample), args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
            processor.saveTree()
        processor.closeInputFile()
        processor.closeOutputFile()
    ### We will treat cross-section normalization separately for diboson, just rename the samples
    shutil.move(f"{BASEDIR}/WZTo3LNu_amcatnlo.root", f"{BASEDIR}/WZ.root")
    shutil.move(f"{BASEDIR}/ZZTo4L_powheg.root", f"{BASEDIR}/ZZ.root")

    ## ttX
    logging.info("Processing ttX...")
    for sample in TTXBKGs:
        input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
        output_path = f"{BASEDIR}/{sample}.root"
        assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)
        for syst in [syst for systset in promptSysts for syst in systset]:
            processor.setInputTree(syst)
            processor.fillOutTree(getSampleAlias(sample), args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
            processor.saveTree()
        processor.closeInputFile()
        processor.closeOutputFile()
    ### We will treat cross-section normalization separately for diboson, just rename the samples
    shutil.move(f"{BASEDIR}/ttWToLNu.root", f"{BASEDIR}/ttW.root")
    shutil.move(f"{BASEDIR}/ttZToLLNuNu.root", f"{BASEDIR}/ttZ.root")
    shutil.move(f"{BASEDIR}/ttHToNonbb.root", f"{BASEDIR}/ttH.root")
    shutil.move(f"{BASEDIR}/tZq.root", f"{BASEDIR}/tZq.root")

    ## others
    logging.info("Processing others...")
    for sample in RAREBKGs:
        input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel}__/PromptSkimmer_{sample}.root"
        output_path = f"{BASEDIR}/{sample}.root"
        assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    
        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)
        for syst in [syst for systset in promptSysts for syst in systset]:
            processor.setInputTree(syst)
            processor.fillOutTree("others", args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
            processor.saveTree()
        processor.closeInputFile()
        processor.closeOutputFile()
    hadd_line = f"hadd -f {BASEDIR}/others.root"
    for sample in RAREBKGs:
        hadd_line += f" {BASEDIR}/{sample}.root"
    os.system(hadd_line)
