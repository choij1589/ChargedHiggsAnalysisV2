#ifndef AMASSFITTER_H
#define AMASSFITTER_H

#include <iostream>
#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <RooRealVar.h>
#include <RooDataSet.h>
#include <RooVoigtian.h>
#include <RooFitResult.h>
#include <RooPlot.h>
#include <vector>

using namespace std;
using namespace ROOT;
using namespace RooFit;

class AmassFitter {
public:
    AmassFitter(const TString &input_path, const TString &output_path);
    ~AmassFitter() = default;

    void fitMass(const double &input_mA, const double &input_low, const double &input_high);
    void saveCanvas(const TString &canvas_path);
    void Close();

    // getters
    RooRealVar *getRooMass() { return roo_mass; }
    RooRealVar *getRooWeight() { return roo_weight; }
    RooRealVar *getRooMA() { return roo_mA; }
    RooRealVar *getRooSigma() { return roo_sigma; }
    RooRealVar *getRooWidth() { return roo_width; }
    RooVoigtian *getRooModel() { return roo_model; }
private:
    TFile *input_file;
    TFile *output_file;
    RooDataSet *roo_data;
    RooRealVar *roo_mass;
    RooRealVar *roo_weight;
    RooRealVar *roo_mA;
    RooRealVar *roo_sigma;
    RooRealVar *roo_width;
    RooVoigtian *roo_model;
    TCanvas *canvas;
};

#endif
