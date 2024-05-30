#include "FitMT.h"

FitMT::FitMT(const TString& era, const TString& hlt, const TString& id, const TString &syst)
    : _ERA(era), HLT(hlt), ID(id), SYST(syst), MT("MT", "MT", 0., 150.) {

    WORKDIR = getenv("WORKDIR") ? TString(getenv("WORKDIR")) : TString("");
    // Set prompt samples
    W = {"WJets_MG"};
    Z = {"DYJets", "DYJets10to50_MG"};
    TT = {"TTLL_powheg", "TTLJ_powheg"};
    ST = {"SingleTop_sch_Lep", "SingleTop_tch_top_Incl", "SingleTop_tch_antitop_Incl", "SingleTop_tW_top_NoFullyHad", "SingleTop_tW_antitop_NoFullyHad"};
    VV = {"WW_pythia", "WZ_pythia", "ZZ_pythia"};
    if (HLT.Contains("Mu")) {
        QCD_MuEnriched = {"QCD_MuEnriched"};
        KEYs = {"QCD_MuEnriched", "Prompt"};
    } else if (HLT.Contains("El")) {
        QCD_EMEnriched = {"QCD_EMEnriched"};
        QCD_bcToE = {"QCD_bcToE"};
        KEYs = {"QCD_EMEnriched", "QCD_bcToE", "Prompt"};
    } else {
        cerr << "ERROR -- FitMT::FitMT() -- Invalid HLT: " << HLT << endl;
        exit(1);
    }

    // Set DATASTREAM
    if (HLT.Contains("Mu")) {
        DATASTREAM = "DoubleMuon";
    } else if (HLT.Contains("El") && _ERA.Contains("2016")) {
        DATASTREAM = "DoubleEG";
    } else if (HLT.Contains("El") && _ERA.Contains("2017")) {
        DATASTREAM = "SingleElectron";
    } else if (HLT.Contains("El") && _ERA.Contains("2018")) {
        DATASTREAM = "EGamma";
    } else {
        cerr << "ERROR -- FitMT::FitMT() -- Invalid HLT: " << HLT << endl;
        exit(1);
    }

    PROMPTs = W;
    PROMPTs.insert(PROMPTs.end(), Z.begin(), Z.end());
    PROMPTs.insert(PROMPTs.end(), TT.begin(), TT.end());
    PROMPTs.insert(PROMPTs.end(), ST.begin(), ST.end());
    PROMPTs.insert(PROMPTs.end(), VV.begin(), VV.end());

    MCSamples = PROMPTs;
    MCSamples.insert(MCSamples.end(), QCD_MuEnriched.begin(), QCD_MuEnriched.end());
    MCSamples.insert(MCSamples.end(), QCD_EMEnriched.begin(), QCD_EMEnriched.end());
    MCSamples.insert(MCSamples.end(), QCD_bcToE.begin(), QCD_bcToE.end());
    histkey = "Inclusive/"+ID+"/"+SYST+"/MTfix";
}

void FitMT::loadHistogram(TFile* f, const TString& sample, const TString& prefix, const TString& key) {
    TString this_histkey = prefix+"/"+histkey;;
    TH1D* h = dynamic_cast<TH1D*>(f->Get(this_histkey));
    if (h) {
        h->SetDirectory(0);
        if (HISTs.find(key) == HISTs.end()) {
            HISTs[key] = (TH1D*)h->Clone("h_"+key);
        } else {
            HISTs[key]->Add(h);
        }
    } else {
        cerr << "ERROR -- FitMT::loadHistogram() -- Cannot find histogram: " << histkey << endl;
        exit(1);
    }
}

void FitMT::prepareTemplates(const TString& prefix) {
    TFile* f = nullptr;

    // load MC
    if (HLT.Contains("Mu")) {
        for (const auto& sample: QCD_MuEnriched) {
            f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
            loadHistogram(f, sample, prefix, "QCD_MuEnriched");
            f->Close();
        }
    } else if (HLT.Contains("El")) {
        for (const auto& sample: QCD_EMEnriched) {
            f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
            loadHistogram(f, sample, prefix, "QCD_EMEnriched");
            f->Close();
        }
        for (const auto& sample: QCD_bcToE) {
            f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
            loadHistogram(f, sample, prefix, "QCD_bcToE");
            f->Close();
        }
    } else {
        cerr << "ERROR -- FitMT::prepareTemplates() -- Invalid HLT: " << HLT << endl;
        exit(1);
    }
    for (const auto& sample: W) {
        f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
        loadHistogram(f, sample, prefix, "Prompt");
        f->Close();
    }
    for (const auto& sample: Z) {
        f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
        loadHistogram(f, sample, prefix, "Prompt");
        f->Close();
    }
    for (const auto& sample: TT) {
        f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
        loadHistogram(f, sample, prefix, "Prompt");
        f->Close();
    }
    for (const auto& sample: ST) {
        f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
        loadHistogram(f, sample, prefix, "Prompt");
        f->Close();
    }
    for (const auto& sample: VV) {
        f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/MeasFakeRateV2_"+sample+".root");
        loadHistogram(f, sample, prefix, "Prompt");
        f->Close();
    }

    double scale = 0.;
    for (const auto& key: KEYs) {
        TH1D* h = HISTs[key];
        RooDataHist* dh = new RooDataHist("dh_"+key, "", MT, Import(*h));
        RooHistPdf* pdf = new RooHistPdf("template_"+key, "", MT, *dh, 0);
        TEMPLATEs[key] = pdf;
        scale += h->Integral();
    }
    for (const auto& key: KEYs) {
        TH1D* h = HISTs[key];
        SCALEs[key] = h->Integral()/scale;
    }
}

void FitMT::prepareCoefficients() {
    for (const auto& key: KEYs) {
        RooRealVar* coef = new RooRealVar("coef_"+key, "", SCALEs[key], 0., 1.);
        COEFs[key] = coef;
    }
}

RooFitResult* FitMT::fit(const TString& prefix) {
    // load data
    RooDataHist* data = nullptr;
    TFile* f = new TFile(WORKDIR+"/SKFlatOutput/MeasFakeRateV2/"+_ERA+"/"+HLT+"__RunSystSimple__/DATA/MeasFakeRateV2_"+DATASTREAM+".root");
    TString this_histkey = prefix+"/"+histkey;
    TH1D* h_data = dynamic_cast<TH1D*>(f->Get(this_histkey));

    if (h_data) {
        h_data->SetDirectory(0);
        data = new RooDataHist("dh_"+DATASTREAM, "", MT, Import(*h_data));
    } else {
        cerr << "ERROR -- FitMT::performFit() -- Cannot find histogram: " << histkey << endl;
        exit(1);
    }
    f->Close();

    prepareTemplates(prefix);
    prepareCoefficients();

    // make model
    RooAddPdf* model = nullptr;
    // Coefficients of W will be given recursively
    if (HLT.Contains("Mu")) {
        model = new RooAddPdf("model", "",
            RooArgList(*TEMPLATEs["QCD_MuEnriched"],
                       *TEMPLATEs["Propmt"]),
            RooArgList(*COEFs["QCD_MuEnriched"]),
            kTRUE);
    } else if (HLT.Contains("El")) {
        model = new RooAddPdf("model", "",
            RooArgList(*TEMPLATEs["QCD_EMEnriched"],
                       *TEMPLATEs["QCD_bcToE"],
                       *TEMPLATEs["Prompt"]),
            RooArgList(*COEFs["QCD_EMEnriched"],
                       *COEFs["QCD_bcToE"]),
            kTRUE);
    } else {
        cerr << "ERROR -- FitMT::performFit() -- Invalid HLT: " << HLT << endl;
        exit(1);
    }
    RooFitResult* fitResult = model->fitTo(*data, Save(), SumW2Error(kTRUE), PrintLevel(-1));
    RooPlot* frame = MT.frame();
    data->plotOn(frame, Name("Data"));
    model->plotOn(frame, Name("Prediction"));
    double chi2 = frame->chiSquare();

    if (HLT.Contains("Mu")) {
        model->plotOn(frame, Name("QCD_MuEnriched"),
                      Components("template_QCD_MuEnriched"),
                      LineColor(kGreen+3), LineStyle(kDashed));
    } else if (HLT.Contains("El")) {
        model->plotOn(frame, Name("QCD_EMEnriched"),
                      Components("template_QCD_EMEnriched"),
                      LineColor(kMagenta), LineStyle(kDashed));
        model->plotOn(frame, Name("QCD_bcToE"),
                      Components("template_QCD_bcToE"),
                      LineColor(kCyan), LineStyle(kDashed));
    }
    model->plotOn(frame, Name("Prompt"),
                  Components("template_Prompt"),
                  LineColor(kOrange+7), LineStyle(kDashed));

    TCanvas* c = new TCanvas("c", "", 800, 800);
    TLegend* l = new TLegend(0.6, 0.6, 0.85, 0.85);
    TLatex* tex = new TLatex();
    l->SetBorderSize(0);
    l->AddEntry(frame->findObject("Data"), "Data", "EP");
    l->AddEntry(frame->findObject("Prediction"), "Prediction", "L");
    if (HLT.Contains("Mu")) {
        l->AddEntry(frame->findObject("QCD_MuEnriched"), "QCD_MuEnriched", "L");
    } else if (HLT.Contains("El")) {
        l->AddEntry(frame->findObject("QCD_EMEnriched"), "QCD_EMEnriched", "L");
        l->AddEntry(frame->findObject("QCD_bcToE"), "QCD_bcToE", "L");
    }
    l->AddEntry(frame->findObject("Prompt"), "Prompt", "L");
    c->cd();
    frame->Draw();
    l->Draw();
    // Draw chi2 with latex
    tex->SetNDC();
    tex->SetTextSize(0.05);
    tex->DrawLatexNDC(0.1, 0.91, "CMS");
    tex->SetTextSize(0.03);
    tex->DrawLatexNDC(0.55, 0.5, "HLT: "+HLT);
    tex->DrawLatexNDC(0.55, 0.45, "ID: "+ID);
    tex->DrawLatexNDC(0.55, 0.4, "SYST: "+SYST);
    tex->DrawLatexNDC(0.55, 0.35, "#chi^{2}/ndof: "+TString::Format("%.2f", chi2));

    const TString outdir = WORKDIR+"/MeasFakeRateV2/plots/"+_ERA+"/"+HLT+"/"+SYST;
    const TString outname = outdir+"/"+prefix+"_"+ID+".png";
    // make directory
    gSystem->mkdir(outdir, kTRUE);
    c->SaveAs(outname);

    // clean up
    delete c;
    delete l;
    delete model;
    delete data;
    delete frame;

    return fitResult;
}
