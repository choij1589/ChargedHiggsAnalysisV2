#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <TFile.h>
#include <TTree.h>

using namespace std;

class Preprocessor {
public:
    Preprocessor(const TString &era, const TString &channel, const TString &datastream);
    ~Preprocessor() = default;
    void setInputFile(const TString &rtfile_path) { inFile = new TFile(rtfile_path, "READ"); }
    void setInputTree(const TString &syst);
    void setOutputFile(const TString &outFileName) { outFile = new TFile(outFileName, "RECREATE"); }
    void closeOutputFile() { outFile->Close(); }
    void closeInputFile() { inFile->Close(); }
    void fillOutTree(const TString &sampleName, const TString &signal, const TString &syst="Central", const bool applyConvSF=false, const bool isTrainedSample=false);
    void saveTree() { outFile->cd(); outTree->Write(); }

    // setters
    void setConvSF(const double &convSF, const double &convSFerr) { this->convSF = convSF; this->convSFerr = convSFerr; }
    void setEra(const TString &era) { this->era = era; }
    void setChannel(const TString &channel) { this->channel = channel; }
    void setDatastream(const TString &datastream) { this->datastream = datastream; }
    
    // getters
    double getConvSF(int sys=0) const { return convSF + double(sys)*convSFerr; }
    TString getEra() const { return era; }
    TString getChannel() const { return channel; }
    TString getDatastream() const { return datastream; }
    // For debugging
    TTree *getOutTree() const { return outTree; }
    TTree *getInTree() const { return inTree; }

private:
    TString era;
    TString channel;
    TString datastream;
    TFile *inFile;
    TFile *outFile;
    TTree *inTree;
    TTree *outTree;

    double convSF, convSFerr;
    double mass, mass1, mass2, scoreX, scoreY, scoreZ, weight;
};

#endif
