#!/usr/bin/env python
import os
import shutil
import logging
import argparse
import ROOT
from itertools import product

logging.basicConfig(level=logging.WARNING)

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--channel", required=True, type=str, help="channel name")
parser.add_argument("--requireBtagged", action="store_true", default=False, help="require b-tagged")
args = parser.parse_args()

# Global Variables
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable is not set.")

CHANNEL = f"{args.channel}__OnlyBtagged__" if args.requireBtagged else f"{args.channel}__"
DATAPROCESSDIR = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/DataPreprocess"
ERAs = ["2016preVFP", "2016postVFP", "2017", "2018"]
SIGNALs = ["MHc-160_MA-85", "MHc-130_MA-90", "MHc-100_MA-95"]
NONPROMPTs = ["TTLL_powheg", "DYJetsToMuMu_MiNNLO"]
DIBOSONs = ["WZTo3LNu_mllmin0p1_powheg", "ZZTo4L_powheg"]
TTZ = ["ttZToLLNuNu", "tZq"]
BACKGROUNDs = NONPROMPTs + DIBOSONs + TTZ

# No. of events to copy
EvtsToCopy = {"signal": [15000, 15000, 30000, 45000],
               "TTLL_powheg": [12000, 12000, 24000, 37000],
               "DYJetsToMuMu_MiNNLO": [3000, 3000, 6000, 8000],
               "WZTo3LNu_mllmin0p1_powheg": [10000, 10000, 20000, 30000],
               "ZZTo4L_powheg": [5000, 5000, 10000, 15000],
               "ttZToLLNuNu": [12000, 12000, 24000, 37000],
               "tZq": [3000, 3000, 6000, 8000]}

# Functions
def create_output_dirs():
    for era in ERAs:
        dir_path = f"dataset/{era}/{CHANNEL}"
        if os.path.exists(dir_path):
            logging.warning(f"{dir_path} already exists. Removing it.")
            shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path)   

def copy_trees(input_file, output_file, n_evts):
    logging.info(f"Copying {input_file} to {output_file}")
    try:
        f = ROOT.TFile.Open(input_file)
        tree = f.Get("Events")
        n_entries = tree.GetEntries()
        if n_evts > n_entries:
            logging.warning(f"Requested events ({n_evts}) exceed available entries ({n_entries}). Adjusting.")
            n_evts = n_entries
        f.Close()

        os.system(f"{WORKDIR}/ParticleNet/libs/copytree {input_file} {n_evts}")
        shutil.move(input_file.replace('.root', '_copy.root'), output_file)
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}")
        
def hadd_files(files, output_file):
    if len(files) == 1:
        shutil.copyfile(files[0], output_file)
    else:
        os.system(f"hadd -f {output_file} {' '.join(files)}")
    

if __name__ == "__main__":
    create_output_dirs()
    for era, signal in product(ERAs, SIGNALs):
        input_file = f"{DATAPROCESSDIR}/{era}/{CHANNEL}/DataPreprocess_TTToHcToWAToMuMu_{signal}.root"
        output_file = f"{WORKDIR}/ParticleNet/dataset/{era}/{CHANNEL}/DataPreprocess_TTToHcToWAToMuMu_{signal}.root"
        n_evts = EvtsToCopy["signal"][ERAs.index(era)]
        copy_trees(input_file, output_file, n_evts)
    
    for era, bkg in product(ERAs, BACKGROUNDs):
        input_file = f"{DATAPROCESSDIR}/{era}/{CHANNEL}/DataPreprocess_{bkg}.root"
        output_file = f"{WORKDIR}/ParticleNet/dataset/{era}/{CHANNEL}/DataPreprocess_{bkg}.root"
        n_evts = EvtsToCopy[bkg][ERAs.index(era)]
        copy_trees(input_file, output_file, n_evts)
        
    for era in ERAs:
        nonprompt_files = [f"dataset/{era}/{CHANNEL}/DataPreprocess_{bkg}.root" for bkg in NONPROMPTs]
        diboson_files = [f"dataset/{era}/{CHANNEL}/DataPreprocess_{bkg}.root" for bkg in DIBOSONs]
        ttZ_file = [f"dataset/{era}/{CHANNEL}/DataPreprocess_{bkg}.root" for bkg in TTZ]

        hadd_files(nonprompt_files, f"dataset/{era}/{CHANNEL}/DataPreprocess_nonprompt.root")
        hadd_files(diboson_files, f"dataset/{era}/{CHANNEL}/DataPreprocess_diboson.root")
        hadd_files(ttZ_file, f"dataset/{era}/{CHANNEL}/DataPreprocess_ttZ.root")        
