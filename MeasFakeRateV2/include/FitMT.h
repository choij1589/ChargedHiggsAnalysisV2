#ifndef FITMT_H
#define FITMT_H

#include <iostream>
#include <vector>
#include <map>
#include <TSystem.h>
#include <TString.h>
#include <TFile.h>
#include <TH1D.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistPdf.h>
#include <RooAddPdf.h>
#include <RooPlot.h>
#include <RooFitResult.h>
#include <RooArgList.h>
#include <TLegend.h>
#include <TCanvas.h>
#include <TLatex.h>
using namespace std;
using namespace RooFit;

class FitMT {
public:
    FitMT(const TString &era, const TString &hlt, const TString &id, const TString &syst);
    RooFitResult* fit(const TString &prefix);

private:
    TString WORKDIR;
    TString _ERA;
    TString HLT;
    TString ID;
    TString SYST;
    TString DATASTREAM;
    vector<TString> QCD_MuEnriched, QCD_EMEnriched, QCD_bcToE;
    vector<TString> W;
    vector<TString> Z;
    vector<TString> TT;
    vector<TString> ST;
    vector<TString> VV;
    vector<TString> PROMPTs;
    vector<TString> MCSamples;
    vector<TString> KEYs;
    TString histkey;

    map<TString, TH1D*> HISTs;
    map<TString, RooHistPdf*> TEMPLATEs;
    map<TString, double> SCALEs;
    map<TString, RooRealVar*> COEFs;
    RooRealVar MT;

    void loadHistogram(TFile *f, const TString& sample, const TString& prefix, const TString& key);
    void prepareTemplates(const TString& prefix);
    void prepareCoefficients();
};

#endif
