#!/usr/bin/env python
import os, shutil
import logging
import argparse
import ROOT

parser = argparse.ArgumentParser(description='Check templates')
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--masspoint", required=True, type=str, help="masspoint")
parser.add_argument("--method", required=True, type=str, help="do ParticleNet optimization")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.getenv("WORKDIR")
BASEDIR = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}"

# Backgrounds to be tested
BACKGROUNDs = ["WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq"]
if args.channel == "SR1E2Mu":
    promptSysts = ["L1Prefire", "PileupReweight",
                   "MuonIDSF", "ElectronIDSF", "TriggerSF",
                   "JetRes", "JetEn", "MuonEn", "ElectronRes", "ElectronEn"]
elif args.channel == "SR3Mu":
    promptSysts = ["L1Prefire", "PileupReweight",
                   "MuonIDSF", "TriggerSF",
                   "JetRes", "JetEn", "MuonEn"]
else:
    raise f"Wrong channel {args.channel}"

# Check histograms for each background and if the norm of any systematics is negative, merge it to others
def checkIntegral(process, systs):
    f = ROOT.TFile.Open(f"{BASEDIR}/shapes_input.precheck.root", "READ")
    nominal = f.Get(process)
    nominalIntegral = nominal.Integral()
    if nominalIntegral < 0:
        logging.info(f"Nominal integral of {process} is negative")
        f.Close()
        return process
    for syst in systs:
        h = f.Get(f"{process}_{syst}Up")
        if h.Integral() < 0:
            logging.info(f"{syst}Up integral of {process} is negative")
            f.Close()
            return process
        h = f.Get(f"{process}_{syst}Down")
        if h.Integral() < 0:
            logging.info(f"{syst}Down integral of {process} is negative")
            f.Close()
            return process
    f.Close()
    return None

def copyTemplates(process):
    logging.info(f"Copy {process}")
    f = ROOT.TFile.Open(f"{BASEDIR}/shapes_input.precheck.root")
    f_out = ROOT.TFile.Open(f"{BASEDIR}/shapes_input.root", "UPDATE")
    
    # find all the keys with "process" in the name
    keys = [key.GetName() for key in f.GetListOfKeys() if process in key.GetName()]
    for key in keys:
        h = f.Get(key); h.SetDirectory(0)
        f_out.cd(); h.Write()
    f.Close()
    f_out.Close()
    

def mergeToOthers(bkgToMerge):
    logging.info("Copy others")
    if len(bkgToMerge) > 0:
        logging.info(f"Merge {bkgToMerge} to others")
    
    f = ROOT.TFile.Open(f"{BASEDIR}/shapes_input.precheck.root")
    f_out = ROOT.TFile.Open(f"{BASEDIR}/shapes_input.root", "UPDATE")
    others_central = f.Get("others"); others_central.SetDirectory(0)

    for bkg in bkgToMerge:
        process_central = f.Get(bkg); process_central.SetDirectory(0)
        others_central.Add(process_central)
    f_out.cd(); others_central.Write()
    
    for syst in promptSysts:
        others = f.Get(f"others_{syst}Up"); others.SetDirectory(0)
        for bkg in bkgToMerge:
            h = f.Get(f"{bkg}_{syst}Up"); h.SetDirectory(0)
            others.Add(h)
        f_out.cd(); others.Write()
        
        others = f.Get(f"others_{syst}Down"); others.SetDirectory(0)
        for bkg in bkgToMerge:
            h = f.Get(f"{bkg}_{syst}Down"); h.SetDirectory(0)
            others.Add(h)
        f_out.cd(); others.Write()
    f.Close()
    f_out.Close()    

if __name__ == "__main__":
    # Back up the original file
    shutil.move(f"{BASEDIR}/shapes_input.root", f"{BASEDIR}/shapes_input.precheck.root")
    bkgToCopy = []
    bkgToMerge = []
    for bkg in BACKGROUNDs:
        if checkIntegral(bkg, promptSysts):
            bkgToMerge.append(bkg)
        else:
            bkgToCopy.append(bkg)

    copyTemplates("data_obs")
    copyTemplates(args.masspoint)
    copyTemplates("nonprompt")
    copyTemplates("conversion")
    for bkg in bkgToCopy:
        copyTemplates(bkg)
    mergeToOthers(bkgToMerge)    
