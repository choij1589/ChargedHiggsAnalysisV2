#include "AmassFitter.h"

AmassFitter::AmassFitter(const TString &input_path, const TString &output_path) {
    input_file = new TFile(input_path, "READ");
    output_file = new TFile(output_path, "RECREATE");
}

void AmassFitter::fitMass(const double &mA, const double &low, const double &high) {
    TTree *tree = static_cast<TTree*>(input_file->Get("Events_Central"));
    
    // Create the output file before creating new TTree
    TTree *copy = new TTree("Events", "");

    double pair1_mass; tree->SetBranchAddress("mass1", &pair1_mass);
    double pair2_mass; tree->SetBranchAddress("mass2", &pair2_mass);
    double weight; tree->SetBranchAddress("weight", &weight); copy->Branch("weight", &weight);
    
    double Amass; copy->Branch("Amass", &Amass);

    for (unsigned int i = 0; i < tree->GetEntries(); i++) {
        tree->GetEntry(i);
        if (pair1_mass < 0. && pair2_mass < 0.) continue;
        
        Amass = fabs(pair1_mass - mA) < fabs(pair2_mass - mA) ? pair1_mass : pair2_mass;
        copy->Fill();
    }

    // Load dataset
    roo_mass = new RooRealVar("Amass", "Amass", low, high);
    roo_weight = new RooRealVar("weight", "weight", 0., 10.);
    roo_data = new RooDataSet("data", "", RooArgSet(*roo_mass, *roo_weight), WeightVar(*roo_weight), Import(*copy));

    roo_mA = new RooRealVar("mA", "mA", mA, low, high);
    roo_sigma = new RooRealVar("sigma", "sigma", 0.1, 0., 3.);
    roo_width = new RooRealVar("width", "width", 0.1, 0., 3.);
    roo_model = new RooVoigtian("model", "model", *roo_mass, *roo_mA, *roo_width, *roo_sigma);
    roo_model->fitTo(*roo_data, SumW2Error(kTRUE));
}

void AmassFitter::saveCanvas(const TString &canvas_path) {
    canvas = new TCanvas("canvas", "canvas", 800, 600);
    RooPlot *plot = roo_mass->frame();
    roo_data->plotOn(plot);
    roo_model->plotOn(plot, LineColor(2));
    plot->Draw();
    canvas->SaveAs(canvas_path);
    canvas->Close();
}

void AmassFitter::Close() {
    input_file->Close();
    output_file->cd();
    roo_data->Write();
    roo_model->Write();
    output_file->Close();
}
