#!/usr/bin/env python3
import os
import ROOT
import argparse
import pandas as pd
import ctypes

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--masspoint", required=True, type=str, help="masspoint")
parser.add_argument("--method", required=True, type=str, help="Baseline / ParticleNet")
args = parser.parse_args()

WORKDIR = os.environ["WORKDIR"]
mA = int(args.masspoint.split("_")[1].split("-")[1])
if args.channel == "Skim1E2Mu":
    promptSysts = ["L1Prefire", 
                   "PileupReweight", 
                   "MuonIDSF", 
                   "ElectronIDSF", 
                   "TriggerSF", 
                   "JetRes", 
                   "JetEn", 
                   "ElectronRes", 
                   "ElectronEn", 
                   "MuonEn"]
if args.channel == "Skim3Mu":
    promptSysts = ["L1Prefire", 
                   "PileupReweight", 
                   "MuonIDSF", 
                   "TriggerSF", 
                   "JetRes", 
                   "JetEn", 
                   "MuonEn"]

class DatacardManager():
    def __init__(self, era, channel, masspoint, method, backgrounds):
        self.signal = masspoint
        self.method = method
        self.csv = None         # for cut and count
        self.rtfile = None      # for shape
        self.backgrounds = []
        if method == "CnC":
            csv_path = f"results/{era}/{channel}__{method}__/{masspoint}/eventRates.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.csv = pd.read_csv(csv_path, index_col="syst")
        else:
            rtfile_path = f"templates/{era}/{channel}/{masspoint}/Shape/{args.method}/shapes_input.root"
            os.makedirs(os.path.dirname(rtfile_path), exist_ok=True)
            self.rtfile = ROOT.TFile.Open(rtfile_path)
        
        for bkg in backgrounds:
            # check if central event rates are positive
            if not self.get_event_rate(bkg) > 0: continue
            self.backgrounds.append(bkg)
        
    def get_event_rate(self, process, syst="Central"):
        if process == "data_obs":
            if self.method == "CnC":
                raise ValueError("[DatacardManager] No event rate for data_obs for method CnC")
            else:
                h = self.rtfile.Get("data_obs")
                return h.Integral()
        else:
            if self.method == "CnC":
                if syst == "stat":  return eval(self.csv.loc["Central", process])[1]
                else:               return eval(self.csv.loc[syst, process])[0]
            else:
                err = ctypes.c_double()
                if syst in ["Central", "stat"]: 
                    if process == "signal": h = self.rtfile.Get(args.masspoint)
                    else:                   h = self.rtfile.Get(process)
                else:                           
                    if process == "signal": h = self.rtfile.Get(f"{args.masspoint}_{syst}")
                    else:                   h = self.rtfile.Get(f"{process}_{syst}")
                content = h.IntegralAndError(1, h.GetNbinsX(), err)
                if syst == "stat": return err.value
                else:              return content
                
    def get_event_statistic(self, process, syst="Central"):
        if self.method == "CnC":
            raise ValueError("[DatacardManager] No support for histogram statistics for CunNCount")
        
        if syst == "Central": 
            if process == "signal": h = self.rtfile.Get(args.masspoint)
            else:                   h = self.rtfile.Get(process)
        else:                 
            if process == "signal": h = self.rtfile.Get(f"{args.masspoint}_{syst}")
            else:                   h = self.rtfile.Get(f"{process}_{syst}")
        return (h.GetMean(), h.GetStdDev())

    def get_event_ratio(self, process, syst):
        if syst == "stat":
            return 1. + self.get_event_rate(process, syst) / self.get_event_rate(process)
        else:
            return 1. + abs(max(self.get_event_rate(process, syst), 0.) / self.get_event_rate(process) - 1.)

    def part1string(self):
        part1string = f"imax\t\t\t1 number of bins\n"
        part1string += f"jmax\t\t\t{len(self.backgrounds)} number of bins\n"
        part1string += f"kmax\t\t\t* number of nuisance parameters\n"
        part1string += "-"*50
        
        if not self.method == "CnC":
            part1string += "\n"
            part1string += "shapes\t*\t*\tshapes_input.root\t$PROCESS\t$PROCESS_$SYSTEMATIC\n"
            part1string += f"shapes\tsignal\t*\tshapes_input.root\t{self.signal}\t{self.signal}_$SYSTEMATIC\n"
            part1string += "-"*50

        return part1string
    
    def part2string(self):
        observation = 0.
        if self.method == "CnC":
            for bkg in self.backgrounds: observation += self.get_event_rate(bkg)
        else:
            observation = self.get_event_rate("data_obs")
        part2string = "bin\t\t\tsignal_region\n"
        part2string += f"observation\t\t{observation:.4f}\n"
        part2string += "-"*50
        return part2string
    
    def part3string(self):
        part3string = "bin\t\t\t" + "signal_region\t" * (len(self.backgrounds)+1) + "\n"
        part3string += "process\t\t\tsignal\t\t"
        for bkg in self.backgrounds: 
            if len(bkg) < 8: part3string += f"{bkg}\t\t"
            else:            part3string += f"{bkg}\t"
        part3string += "\n"
        
        part3string += "process\t\t\t0\t\t"
        for idx in range(1, len(self.backgrounds)+1): part3string += f"{idx}\t\t"
        part3string += "\n"
        
        if self.method == "CnC":
            part3string += "rate\t\t\t"
            part3string += f"{self.get_event_rate('signal'):.2f}\t\t"
            for bkg in self.backgrounds:
                part3string += f"{self.get_event_rate(bkg):.2f}\t\t"
        else:
            part3string += f"rate\t\t\t-1\t\t"
            part3string += "-1\t\t" * len(self.backgrounds)
        part3string += "\n"
        part3string += "-"*50
        return part3string
    
    def autoMCstring(self, threshold):
        if self.method == "CnC":
            print("[Datacardmanager] autoMCstat only supports for the shape method")
        return f"signal_region\tautoMCStats\t{threshold}"

    def syststring(self, syst, alias=None, sysType=None, value=None, skip=None, denoteEra=False):
        if syst == "Nonprompt" and (not "nonprompt" in self.backgrounds): return ""
        if syst == "Conversion" and (not "conversion" in self.backgrounds): return ""

        # set alias
        if alias is None: alias = syst
        if denoteEra:
            if args.era == "2016preVFP": alias = f"{alias}_16a"
            elif args.era == "2016postVFP": alias = f"{alias}_16b"
            elif args.era == "2017": alias = f"{alias}_17"
            elif args.era == "2018": alias = f"{alias}_18"
            else:
                raise ValueError(f"[DatacardManager] What era is {args.era}?")

        # check type
        if self.method == "CnC":
            sysType = "lnN"
        elif sysType is None:  # shape and do denoted type
            # if at least one source is negative, use lnN
            islnN = False
            for process in ["signal"]+self.backgrounds:
                if process in skip: continue
                rate_up = self.get_event_rate(process, f"{syst}Up")
                rate_down = self.get_event_rate(process, f"{syst}Down")
                if not (rate_up >0. and rate_down > 0.): islnN = True

            # now check the mean and stddev
            if not islnN:
                for process in ["signal"]+self.backgrounds:
                    if process in skip: continue
                    mean, stddev = self.get_event_statistic(process)
                    mean_up, stddev_up = self.get_event_statistic(process, f"{syst}Up")
                    mean_down, stddev_down = self.get_event_statistic(process, f"{syst}Down")

                    # if mean & stddev within 0.5%, only vary normalization
                    if stddev < 10e-6:                              continue
                    if abs(mean - mean_up)/mean > 0.005:            islnN = False; break
                    if abs(mean - mean_down)/mean > 0.005:          islnN = False; break
                    if abs(stddev-stddev_up)/stddev > 0.005:        islnN = False; break
                    if abs(stddev-stddev_down)/stddev > 0.005:      islnN = False; break
                
			    # final check for nonprompt & conversion
                #if syst == "Nonprompt":
                #    rate_up = self.get_event_rate("nonprompt", "NonpromptUp")
                #    rate_down = self.get_event_rate("nonprompt", "NonpromptDown")
                #    if not (rate_up > 0. and rate_down > 0.): islnN = True
                #if syst == "Conversion":
                #    rate_up = self.get_event_rate("conversion", "ConversionUp")
                #    rate_down = self.get_event_rate("conversion", "ConversionDown")
                #    if not (rate_up > 0. and rate_down > 0.): islnN = True
            if islnN: sysType = "lnN"
            else:     sysType = "shape"
        else:
            pass

        syststring = f"{alias}\t\t{sysType}\t" if len(alias) < 8 else f"{alias}\t{sysType}\t"
        if sysType == "lnN":
            for process in ["signal"]+self.backgrounds:
                if process in skip: syststring += "-\t\t"
                elif value is None:
                    if syst == "stat":
                        ratio = self.get_event_ratio(process, "stat")
                    else:
                        ratio = max(self.get_event_ratio(process, f"{syst}Up"), self.get_event_ratio(process, f"{syst}Down"))
                    syststring += f"{ratio:.3f}\t\t"
                else:
                    syststring += f"{value:.3f}\t\t"
        elif sysType == "shape":
            for process in ["signal"]+self.backgrounds:
                if process in skip: syststring += "-\t\t"
                else:               syststring += "1\t\t"
        else:
            print(f"[DatacardManager] What type is {sysType}?")
            raise(ValueError)

        return syststring

if __name__ == "__main__":
    # Check backgrounds
    input_file = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}/shapes_input.root"
    if not os.path.exists(input_file):
        print(f"[DatacardManager] {input_file} does not exist")
        raise ValueError
    f = ROOT.TFile.Open(input_file)
    # check all the prompt backgrounds are there
    keys = [key.GetName() for key in f.GetListOfKeys()]
    promptBkgs = ["WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "others"]
    for bkg in ["WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "others"]:
        if bkg not in keys:
            print(f"[DatacardManager] {bkg} is not in the input file")
            promptBkgs.remove(bkg)
    
    manager = DatacardManager(args.era, args.channel, args.masspoint, args.method, ["nonprompt", "conversion"]+promptBkgs)

    print("# signal xsec scaled to be 5 fb")
    if args.method == "CnC":
        print("WARNING!!!! Systematics not properly parsed for CnC")
        print(manager.part1string())
        print(manager.part2string())
        print(manager.part3string())
        print(manager.syststring(syst="lumi_13TeV", value=1.025, skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="stat", alias="norm_signal", skip=["nonprompt", "conversion", "diboson", "ttX", "others"]))
        print(manager.syststring(syst="stat", alias="norm_diboson", skip=["signal", "nonprompt", "conversion", "ttX", "others"]))
        print(manager.syststring(syst="stat", alias="norm_ttX", skip=["signal", "nonprompt", "conversion", "diboson", "others"]))
        print(manager.syststring(syst="stat", alias="norm_others", skip=["signal", "nonprompt", "conversion", "diboson", "ttX"]))
        print(manager.syststring(syst="L1Prefire", alias="l1prefire", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="PileupReweight", alias="pileup", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="MuonIDSF", alias="idsf_muon", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="DblMuTrigSF", alias="trig_dblmu", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="JetRes", alias="res_jet", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="JetEn", alias="en_jet", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="MuonEn", alias="en_muon", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="Nonprompt", alias="nonprompt", skip=["signal", "conversion", "diboson", "ttX", "others"]))
        print(manager.syststring(syst="Conversion", alias="conversion", skip=["signal", "nonprompt", "diboson", "ttX", "others"]))
    else:
        print(manager.part1string())
        print(manager.part2string())
        print(manager.part3string())
        print(manager.autoMCstring(threshold=10))
        print(manager.syststring(syst="lumi_13TeV", sysType="lnN", value=1.025, skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="L1Prefire", sysType="lnN", skip=["nonprompt", "conversion"], denoteEra=True))
        print(manager.syststring(syst="PileupReweight", sysType="lnN", skip=["nonprompt", "conversion"]))
        if args.channel == "SR1E2Mu":
            print(manager.syststring(syst="ElectronIDSF", sysType="lnN", skip=["nonprompt", "conversion"]))
            print(manager.syststring(syst="MuonIDSF", sysType="lnN", skip=["nonprompt", "conversion"]))
            print(manager.syststring(syst="TriggerSF", sysType="lnN", skip=["nonprompt", "conversion"]))
            print(manager.syststring(syst="ElectronRes", skip=["nonprompt", "conversion"]))
            print(manager.syststring(syst="ElectronEn", skip=["nonprompt", "conversion"]))
        if args.channel == "SR3Mu":
            print(manager.syststring(syst="MuonIDSF", sysType="lnN", skip=["nonprompt", "conversion"]))
            print(manager.syststring(syst="TriggerSF", sysType="lnN", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="JetRes", skip=["nonprompt", "conversion"], denoteEra=True))
        print(manager.syststring(syst="JetEn", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="MuonEn", skip=["nonprompt", "conversion"]))
        print(manager.syststring(syst="PDF", sysType="shape", skip=["nonprompt", "conversion"]+promptBkgs))
        print(manager.syststring(syst="Scale", sysType="shape", skip=["nonprompt", "conversion"]+promptBkgs))
        print(manager.syststring(syst="PS", sysType="shape", skip=["nonprompt", "conversion"]+promptBkgs))
        print(manager.syststring(syst="Nonprompt", sysType="lnN", value=1.3, skip=["signal", "conversion"]+promptBkgs, denoteEra=True))
        print(manager.syststring(syst="Conversion", sysType="lnN", value=1.2, skip=["signal", "nonprompt"]+promptBkgs, denoteEra=True))
        if "WZ" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("WZ")
            print(manager.syststring(syst="norm_WZ", sysType="lnN", value=1.12, skip=["signal", "nonprompt", "conversion"]+tempBkg))
        if "ZZ" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("ZZ")
            print(manager.syststring(syst="norm_ZZ", sysType="lnN", value=1.064, skip=["signal", "nonprompt", "conversion"]+tempBkg))
        if "ttW" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("ttW")
            print(manager.syststring(syst="norm_ttW", sysType="lnN", value=1.119, skip=["signal", "nonprompt", "conversion"]+tempBkg))
        if "ttZ" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("ttZ")
            print(manager.syststring(syst="norm_ttZ", sysType="lnN", value=1.133, skip=["signal", "nonprompt", "conversion"]+tempBkg))
        if "ttH" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("ttH")
            print(manager.syststring(syst="norm_ttH", sysType="lnN", value=1.1, skip=["signal", "nonprompt", "conversion"]+tempBkg))
        if "tZq" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("tZq")
            print(manager.syststring(syst="norm_tZq", sysType="lnN", value=1.052, skip=["signal", "nonprompt", "conversion"]+tempBkg))
        if "others" in promptBkgs:
            tempBkg = promptBkgs.copy(); tempBkg.remove("others")
            print(manager.syststring(syst="norm_others", sysType="lnN", value=1.5, skip=["signal", "nonprompt", "conversion"]+tempBkg))
