#!/usr/bin/env python
import os, shutil
import logging
import argparse
import numpy as np
import pandas as pd
import ROOT
import pickle

from array import array
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--masspoint", required=True, type=str, help="masspoint")
parser.add_argument("--method", required=True, type=str, help="do ParticleNet optimization")
parser.add_argument("--update", action="store_true", default=False, help="update GBDT score")
parser.add_argument("--nfold", type=int, default=5, help="nfold")
parser.add_argument("--early_stopping_rounds", type=int, default=10, help="early stopping rounds")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.getenv("WORKDIR")
BASEDIR = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}"
BACKGROUNDs = ["nonprompt", "conversion", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "others"]

# If at least one systematic's norm < 0, it means that the rate is really small.
# In this case, we merge the process to "others"
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

theorySysts = [
        ("AlpS_up", "AlpS_down"),
        ("AlpSfact_up", "AlpSfact_down"),
        tuple([f"PDFReweight_{i}" for i in range(100)]),
        tuple([f"ScaleVar_{i}" for i in [0, 1, 2, 3, 4, 6, 8]]),
        tuple([f"PSVar_{i}" for i in range(4)])
        ]

ROOT.gInterpreter.Declare("""
double get_final_score(int fold, double score, double central_score) {
    if (fold == -999) return central_score;
    else              return score;
}
""")

# helper functions
## Fit A mass
def getFitResult(input_path, output_path, mA):
    fitter = ROOT.AmassFitter(input_path, output_path)
    fitter.fitMass(mA, mA-20., mA+20.)
    fitter.saveCanvas(f"{BASEDIR}/fit_result.png")
    mA = fitter.getRooMA().getVal()
    width = fitter.getRooWidth().getVal()
    sigma = fitter.getRooSigma().getVal()
    fitter.Close()
    return mA, width, sigma

## Gradient Boosting Optimization
def loadDataset(process, mA, width, sigma):
    events = {}
    for i in range(args.nfold):
        events[i] = []
    source = ROOT.TFile(f"{WORKDIR}/SignalRegionStudyV1/samples/{args.era}/{args.channel.replace('SR', 'Skim')}/{args.masspoint}/{args.method}/{process}.root")
    tree = source.Get(f"Central")
    window = 5*width+3*sigma
    for evt in tree:
        if evt.fold not in range(args.nfold):
            raise ValueError(f"Wrong fold {evt.fold}")
        condition = (mA - window) < evt.mass1 < (mA+window) or (mA - window) < evt.mass2 < (mA+window)
        if not condition:
            continue
        
        events[evt.fold].append([evt.scoreX, evt.scoreY, evt.scoreZ, evt.weight, int(process == args.masspoint)])
    source.Close()
    return events

def evalSensitivity(y_true, y_pred, weights, threshold=0.):
    signal_mask = (y_true == 1) & (y_pred > threshold)
    background_mask = (y_true == 0) & (y_pred > threshold)

    S = np.sum(weights[signal_mask])
    B = np.sum(weights[background_mask])
    
    return 0. if B <= 0. else np.sqrt(2*((S+B)*np.log(1+S/B)-S))

def trainModels(events_sig, events_bkg):
    models = {}
    for fold_idx in range(args.nfold):
        # fold_idx is the index of the test set, 
        # (fold_idx+1)%args.nfold is the index of the validation set
        # and others are the index of the training set
        train_sig = [events_sig[f].copy() for f in range(args.nfold) if (f not in [fold_idx, (fold_idx+1)%args.nfold])]
        train_bkg = [events_bkg[f].copy() for f in range(args.nfold) if (f not in [fold_idx, (fold_idx+1)%args.nfold])]
        valid_sig = events_sig[(fold_idx+1)%args.nfold].copy()
        valid_bkg = events_bkg[(fold_idx+1)%args.nfold].copy()
        
        # Normalize background weights to match signal weight
        sum_sig = np.sum([np.sum(evt[:,3]) for evt in train_sig])
        sum_bkg = np.sum([np.sum(evt[:,3]) for evt in train_bkg])
        ratio = sum_sig / sum_bkg
        
        for data in train_bkg:
            data[:,3] *= ratio
        valid_bkg[:,3] *= ratio
        
        train_data = shuffle(np.vstack(train_sig + train_bkg), random_state=42)
        valid_data = np.vstack([valid_sig, valid_bkg])
        
        X_train, w_train, y_train = train_data[:, :3], train_data[:, 3], train_data[:, 4]
        X_valid, w_valid, y_valid = valid_data[:, :3], valid_data[:, 3], valid_data[:, 4]
        
        # Train the classifiers
        model = GradientBoostingClassifier(
                    n_estimators=1, 
                    warm_start=True, 
                    max_depth=3, 
                    learning_rate=0.1, 
                    random_state=42)
        val_losses = []
        min_val_loss = np.inf
        for i in range(1, 101):
            model.set_params(n_estimators=i)
            model.fit(X_train, y_train, sample_weight=w_train)
            
            y_pred = model.predict_proba(X_valid)
            val_loss = log_loss(y_valid, y_pred)
            val_losses.append(val_loss)
            
            # check for early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            
            if patience >= args.early_stopping_rounds:
                print(f"Early stopping at {i}th iteration")
                break
    
        models[fold_idx] = model
    return models

def getOptimizedThreshold(models, events_sig, events_bkg):
    # Merge all test results
    val_preds = []
    val_labels = []
    val_weights = []
    for fold_idx in range(args.nfold):
        model = models[fold_idx]
        test_sig = events_sig[fold_idx]
        test_bkg = events_bkg[fold_idx]
        
        testset = np.vstack([test_sig, test_bkg])
        X_test, w_test, y_test = testset[:, :3], testset[:, 3], testset[:, 4]
        y_pred = model.predict_proba(X_test)[:, 1]
        val_preds.append(y_pred)
        val_labels.append(y_test)
        val_weights.append(w_test)
    
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)
    val_weights = np.concatenate(val_weights)
    
    thresholds = np.linspace(0, 1, 101)
    sensitivities = [evalSensitivity(val_labels, val_preds, val_weights, threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(sensitivities)]
    initial_sensitivity = sensitivities[0]
    max_sensitivity = np.max(sensitivities)
    
    return best_threshold, initial_sensitivity, max_sensitivity

def plotGBDTOutput(models, events_sig, events_bkg, best_threshold, improvement):
    h_sig = ROOT.TH1D("signal", "", 100, 0., 1.)
    h_bkg = ROOT.TH1D("background", "", 100, 0., 1.)
    for fold in range(args.nfold):
        model = models[fold]
        y_pred = model.predict_proba(events_sig[fold][:, :3])[:, 1]
        w_pred = events_sig[fold][:, 3]
        for score, weight in zip(y_pred, w_pred):
            h_sig.Fill(score, weight)
        
        y_pred = model.predict_proba(events_bkg[fold][:, :3])[:, 1]
        w_pred = events_bkg[fold][:, 3]
        for score, weight in zip(y_pred, w_pred):
            h_bkg.Fill(score, weight)
    
    h_sig.Scale(1./h_sig.Integral()); h_sig.SetStats(0)
    h_bkg.Scale(1./h_bkg.Integral()); h_bkg.SetStats(0)
    
    h_sig.SetLineColor(ROOT.kRed)
    h_sig.SetLineWidth(2)
    h_bkg.SetLineColor(ROOT.kGray+2)
    h_bkg.SetLineWidth(2)
    h_sig.GetXaxis().SetTitle("GBDT score")
    h_sig.GetYaxis().SetTitle("A.U.")
    h_sig.GetYaxis().SetRangeUser(0., 2.*max(h_sig.GetMaximum(), h_bkg.GetMaximum()))

    line = ROOT.TLine(best_threshold, 0., best_threshold, 0.5*h_sig.GetMaximum())
    line.SetLineColor(ROOT.kBlack) 
    line.SetLineWidth(3)

    l = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    l.AddEntry(h_sig, "Signal", "lep")
    l.AddEntry(h_bkg, "Background", "lep")
    
    latex = ROOT.TLatex()
    
    c = ROOT.TCanvas("c", "c", 800, 600)
    c.cd()
    h_sig.Draw("hist&e")
    h_bkg.Draw("hist&e&same")
    line.Draw("same")
    l.Draw()
    # CMS
    latex.SetTextFont(61)
    latex.SetTextSize(0.05)
    latex.DrawLatexNDC(0.1, 0.905, "CMS")
    # Work in progress
    latex.SetTextFont(52)
    latex.SetTextSize(0.04)
    latex.DrawLatexNDC(0.2, 0.905, "Work in progress")
    # Lumi
    LumiInfo = {    # /fb
        "2016preVFP": 19.5,
        "2016postVFP": 16.8,
        "2017": 41.5,
        "2018": 59.8
    }
    lumiString = "L_{int} ="+f" {LumiInfo[args.era]}"+" fb^{-1} (13TeV)"
    if args.channel == "SR1E2Mu":
        channel = "1e2#mu"
    elif args.channel == "SR3Mu":
        channel = "3#mu"
    else:
        raise ValueError(f"Wrong channel {args.channel}")
    latex.SetTextFont(42)
    latex.SetTextSize(0.03)
    latex.DrawLatexNDC(0.7, 0.905, lumiString)
    # Information, channel, masspoint, threshold and improvement
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatexNDC(0.15, 0.8, f"channel: {channel}")
    latex.DrawLatexNDC(0.15, 0.76, f"masspoint: {args.masspoint}")
    latex.DrawLatexNDC(0.15, 0.72, f"threshold: {best_threshold:.2f}")
    latex.DrawLatexNDC(0.15, 0.68, f"improvement: {improvement*100:.3f}%")
    
    c.SaveAs(f"{BASEDIR}/gbdt_output.png")
    
def updateScore(process, syst, models):
    logging.info(f"Updating GBDT score for {process} {syst}")
    f = ROOT.TFile.Open(f"{WORKDIR}/SignalRegionStudyV1/samples/{args.era}/{args.channel.replace('SR', 'Skim')}/{args.masspoint}/{args.method}/{process}.root", "UPDATE")
    tree = f.Get(syst)
    
    scores = np.zeros(tree.GetEntries(), dtype=float)
    
    for i, evt in enumerate(tree):
        if evt.fold == -999:
            scores[i] = -999.
        else:
            model = models[evt.fold]
            scores[i] = model.predict_proba([[evt.scoreX, evt.scoreY, evt.scoreZ]])[0, 1]
    
    score_array = array('d', [0.])
    score_branch = tree.Branch("score", score_array, "score/D")
    
    for i in range(tree.GetEntries()):
        score_array[0] = scores[i]
        tree.GetEntry(i)
        score_branch.Fill()
    
    f.cd()
    tree.Write("", ROOT.TObject.kOverwrite)
    f.Close() 

def getHist(process, mA, width, sigma, threshold=-999., syst="Central"):
    logging.debug(process, syst)
    file_path = f"{WORKDIR}/SignalRegionStudyV1/samples/{args.era}/{args.channel.replace('SR', 'Skim')}/{args.masspoint}/{args.method}/{process}.root"
    hist_name = process if syst == "Central" else f"{process}_{syst}"
    hist_range = (15, mA - 5 * width - 3 * sigma, mA + 5 * width + 3 * sigma)
    tree_name = syst
    central_name = "Central"
    
    chain = ROOT.TChain(tree_name)
    chain.Add(file_path)
    
    central_chain = ROOT.TChain(central_name)
    central_chain.Add(file_path)
    chain.AddFriend(central_chain, "central")
    
    rdf = ROOT.RDataFrame(chain)
    if args.method == "ParticleNet":
        rdf = rdf.Define("central_score", "central.score")
        rdf = rdf.Define("final_score", "get_final_score(fold, score, central_score)")
        rdf = rdf.Filter(f"final_score >= {threshold}")

    # Fill histograms based on the channel
    if args.channel == "SR1E2Mu":
        hist = rdf.Histo1D((hist_name, "", *hist_range), "mass1", "weight").GetValue()
    elif args.channel == "SR3Mu":
        # Create histograms for mass1 and mass2 separately
        h1 = rdf.Histo1D(("h1", "", *hist_range), "mass1", "weight")
        h2 = rdf.Histo1D(("h2", "", *hist_range), "mass2", "weight")
        # Add the two histograms
        hist = h1.Clone(hist_name)
        hist.Add(h2.GetValue())
    else:
        raise ValueError(f"Unsupported channel: {args.channel}")

    hist.SetDirectory(0)
    return hist

def update():
    events_sig = loadDataset(args.masspoint, mA, width, sigma)
    events_bkg = {}
    for i in range(args.nfold):
        events_bkg[i] = []
    for process in BACKGROUNDs:
        events_temp = loadDataset(process, mA, width, sigma)
        for i in range(args.nfold):
            events_bkg[i] += events_temp[i]
    for i in range(args.nfold):
        events_sig[i] = np.array(events_sig[i])
        events_bkg[i] = np.array(events_bkg[i])
    
    models = trainModels(events_sig, events_bkg)
    # save models
    with open(f"{BASEDIR}/models.pkl", "wb") as f:
        pickle.dump(models, f)
        
    # load models
    with open(f"{BASEDIR}/models.pkl", "rb") as f:
        models = pickle.load(f)
        
    best_threshold, initial_sensitivity, max_sensitivity = getOptimizedThreshold(models, events_sig, events_bkg)
    improvement = (max_sensitivity/initial_sensitivity-1)
    logging.info(f"Best threshold: {best_threshold}")
    logging.info(f"Initial sensitivity: {initial_sensitivity}")
    logging.info(f"Max sensitivity: {max_sensitivity}")    
    logging.info(f"Improved by {improvement*100:.3f}%")
    plotGBDTOutput(models, events_sig, events_bkg, best_threshold, improvement)
    result = pd.DataFrame({"process": [args.masspoint], "threshold": [best_threshold], "initial_sensitivity": [initial_sensitivity], "max_sensitivity": [max_sensitivity], "improvement": [improvement]})
    result.to_csv(f"{BASEDIR}/threshold.csv", index=False)
    
    updateScore(args.masspoint, "Central", models)
    for syst in promptSysts:
        updateScore(args.masspoint, f"{syst}Up", models)
        updateScore(args.masspoint, f"{syst}Down", models)
    for i in range(100):
        updateScore(args.masspoint, f"PDFReweight_{i}", models)
        
    for i in [0, 1, 2, 3, 4, 6, 8]:
        updateScore(args.masspoint, f"ScaleVar_{i}", models)
        
    for i in range(4):
        updateScore(args.masspoint, f"PSVar_{i}", models)
    
    updateScore("nonprompt", "Central", models)
    updateScore("conversion", "Central", models)
    
    for process in ["WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "others"]:
        updateScore(process, "Central", models)
        for syst in promptSysts:
            updateScore(process, f"{syst}Up", models)
            updateScore(process, f"{syst}Down", models)
    
    return models, best_threshold

if __name__ == "__main__":
    if args.method == "ParticleNet":
        if args.update:
            if os.path.exists(BASEDIR): shutil.rmtree(BASEDIR)
            os.makedirs(BASEDIR)
        else:
            assert os.path.exists(BASEDIR), f"{BASEDIR} should exist with updated GBDT models"
    else:
        if os.path.exists(BASEDIR): shutil.rmtree(BASEDIR)
        os.makedirs(BASEDIR)
    
    mA = float(args.masspoint.split("_")[1].split("-")[1])
    
    # fit
    input_path = f"{WORKDIR}/SKFlatOutput/Run2UltraLegacy_v3/PromptSkimmer/{args.era}/{args.channel.replace('SR', 'Skim')}__RunTheoryUnc__/PromptSkimmer_TTToHcToWAToMuMu_{args.masspoint}.root"
    output_path = f"{BASEDIR}/fit_result.root"
    mA, width, sigma = getFitResult(input_path, output_path, mA)
    hist_range = (15, mA - 5 * width - 3 * sigma, mA + 5 * width + 3 * sigma)
    models = None
    best_threshold = -999. 
    
    if args.method == "ParticleNet":
        if args.update: 
            models, best_threshold = update() 
        else: 
            with open(f"{BASEDIR}/models.pkl", "rb") as f: 
                models = pickle.load(f)
            csv = pd.read_csv(f"{BASEDIR}/threshold.csv")
            best_threshold = float(csv.loc[0, 'threshold'])
        
    f = ROOT.TFile(f"{BASEDIR}/shapes_input.root", "RECREATE")
    data_obs = ROOT.TH1D("data_obs", "", *hist_range)
    
    logging.info(f"Processing {args.masspoint}")
    central = getHist(args.masspoint, mA, width, sigma, best_threshold); f.cd(); central.Write()
    for syst in promptSysts:
        hist = getHist(args.masspoint, mA, width, sigma, best_threshold, f"{syst}Up"); f.cd(); hist.Write()
        hist = getHist(args.masspoint, mA, width, sigma, best_threshold, f"{syst}Down"); f.cd(); hist.Write()
        
    # Make up / down histograms for PDF variations
    # first make histograms for each PDF variation, then make up / down histograms
    hists_pdf = []
    for i in range(100):
        hists_pdf.append(getHist(args.masspoint, mA, width, sigma, best_threshold, f"PDFReweight_{i}"))
    # calculate RMS of PDF variations
    pdf_up = central.Clone(f"{args.masspoint}_PDFUp")
    pdf_down = central.Clone(f"{args.masspoint}_PDFDown")
    for i in range(1, central.GetNbinsX()+1):
        bin_values = np.array([hist.GetBinContent(i) for hist in hists_pdf])
        rms = np.std(bin_values, ddof=1)
        pdf_up.SetBinContent(i, central.GetBinContent(i) + rms)
        pdf_down.SetBinContent(i, central.GetBinContent(i) - rms)
    f.cd(); pdf_up.Write(); pdf_down.Write()

    # Make up / down histograms for scale variations
    hists_scale = []
    for i in [0, 1, 2, 3, 4, 6, 8]:
        hists_scale.append(getHist(args.masspoint, mA, width, sigma, best_threshold, f"ScaleVar_{i}"))
    # calculate min/max of scale variations
    scale_up = central.Clone(f"{args.masspoint}_ScaleUp")
    scale_down = central.Clone(f"{args.masspoint}_ScaleDown")
    for i in range(1, central.GetNbinsX()+1):
        bin_values = np.array([hist.GetBinContent(i) for hist in hists_scale])
        scale_up.SetBinContent(i, np.max(bin_values))
        scale_down.SetBinContent(i, np.min(bin_values))
    f.cd(); scale_up.Write(); scale_down.Write()

    # Make up / down histograms for PS variations
    hists_ps = []
    for i in range(4):
        hists_ps.append(getHist(args.masspoint, mA, width, sigma, best_threshold, f"PSVar_{i}"))
    # calculate min/max of PS variations
    ps_up = central.Clone(f"{args.masspoint}_PSUp")
    ps_down = central.Clone(f"{args.masspoint}_PSDown")
    for i in range(1, central.GetNbinsX()+1):
        bin_values = np.array([hist.GetBinContent(i) for hist in hists_ps])
        ps_up.SetBinContent(i, np.max(bin_values))
        ps_down.SetBinContent(i, np.min(bin_values))
    f.cd(); ps_up.Write(); ps_down.Write()

    logging.info("Processing nonprompt")
    hist = getHist("nonprompt", mA, width, sigma, best_threshold); data_obs.Add(hist); f.cd(); hist.Write()

    logging.info("Processing conversion")
    hist = getHist("conversion", mA, width, sigma, best_threshold); data_obs.Add(hist); f.cd(); hist.Write()

    logging.info("Processing diboson")
    for process in ["WZ", "ZZ"]:
        hist = getHist(process, mA, width, sigma, best_threshold); data_obs.Add(hist); f.cd(); hist.Write()
        for syst in promptSysts:
            hist = getHist(process, mA, width, sigma, best_threshold, f"{syst}Up"); f.cd(); hist.Write()
            hist = getHist(process, mA, width, sigma, best_threshold, f"{syst}Down"); f.cd(); hist.Write()

    logging.info("Processing ttX")
    for process in ["ttW", "ttZ", "ttH", "tZq"]:
        hist = getHist(process, mA, width, sigma, best_threshold); data_obs.Add(hist); f.cd(); hist.Write()
        for syst in promptSysts:
            hist = getHist(process, mA, width, sigma, best_threshold, f"{syst}Up"); f.cd(); hist.Write()
            hist = getHist(process, mA, width, sigma, best_threshold, f"{syst}Down"); f.cd(); hist.Write()

    logging.info("Processing others")
    hist = getHist("others", mA, width, sigma, best_threshold); data_obs.Add(hist); f.cd(); hist.Write()
    for syst in promptSysts:
        hist = getHist("others", mA, width, sigma, best_threshold, f"{syst}Up"); f.cd(); hist.Write()
        hist = getHist("others", mA, width, sigma, best_threshold, f"{syst}Down"); f.cd(); hist.Write()

    f.cd(); data_obs.Write()
    f.Close()
