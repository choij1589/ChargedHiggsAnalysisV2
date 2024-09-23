#include "Preprocessor.h"

Preprocessor::Preprocessor(const TString &era,
                           const TString &channel,
                           const TString &datastream)
    : era(era), channel(channel), datastream(datastream) {}

void Preprocessor::setInputTree(const TString &syst) {
    inTree = static_cast<TTree*>(inFile->Get("Events_"+syst));
}

void Preprocessor::fillOutTree(const TString &sampleName, const TString &signal, const TString &syst, const bool applyConvSF, const bool isTrainedSample) {
    outTree = new TTree(sampleName+"_"+syst, "");
    outTree->Branch("mass", &mass);
    outTree->Branch("mass1", &mass1);
    outTree->Branch("mass2", &mass2);
    if (isTrainedSample) {
        outTree->Branch("scoreX", &scoreX);
        outTree->Branch("scoreY", &scoreY);
        outTree->Branch("scoreZ", &scoreZ);
    }
    outTree->Branch("weight", &weight);
    
    // Set branch addresses
    inTree->SetBranchAddress("mass1", &mass1);
    inTree->SetBranchAddress("mass2", &mass2);
    if (isTrainedSample) {
        inTree->SetBranchAddress("score_" + signal + "_vs_nonprompt", &scoreX);
        inTree->SetBranchAddress("score_" + signal + "_vs_diboson", &scoreY);
        inTree->SetBranchAddress("score_" + signal + "_vs_ttZ", &scoreZ);
    }

    inTree->SetBranchAddress("weight", &weight);

    // Loop over tree entries
    for (unsigned int entry = 0; entry < inTree->GetEntries(); ++entry) {
        inTree->GetEntry(entry);

        // Signal cross-section scaling to 5 fb
        if (sampleName.Contains("MA")) {
            weight /= 3.0;
        } else if (applyConvSF) {
            weight *= getConvSF();
        }


        // Process based on the channel
        if (channel.Contains("1E2Mu")) {
            mass = mass1;
            outTree->Fill();
        } else if (channel.Contains("3Mu")) {
            mass = mass1;
            outTree->Fill();
            mass = mass2;
            outTree->Fill();
        } else {
            cerr << "Wrong channel: " << channel << endl;
            exit(EXIT_FAILURE);
        }
    }
}
