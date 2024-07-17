#include <iostream>
#include <string>
#include <TFile.h>
#include <TTree.h>
using namespace std;

int main(int argc, char** argv) {

    const TString inputfile = (TString)argv[1];
    const unsigned int nEvents = stoi(argv[2]);

    cout << "@@@@ inputfile: " << inputfile << endl;
    cout << "@@@@ clone " << nEvents << endl;

    TFile *f = new TFile(inputfile);
    TTree *oldtree = (TTree*)f->Get("Events");
    const unsigned int  nEntries = oldtree->GetEntries();

    /*
    const unsigned int whole = nEntries / nfiles;
    const unsigned int remainder = nEntries % nfiles;

    cout << "@@@@ nEntries: " << nEntries << endl;
    cout << "@@@@ each file contains " << whole << " events" << endl;
    cout << "@@@@ the last file contains " << whole+remainder << " events" << endl;
    */


    TString outfile_name = inputfile;
    TString s_replace = "_copy.root";
    outfile_name.ReplaceAll(".root", s_replace);

    TFile* newfile = new TFile(outfile_name, "recreate");
    TTree* newtree = oldtree->CloneTree(0);

    for (unsigned int entry = 0; entry < nEvents; entry++) {
        oldtree->GetEntry(entry);
        newtree->Fill();
    }
    newfile->cd();
    newtree->Write();

    cout << "@@@@ copied " << newtree->GetEntries() << " events to " << outfile_name << endl;
    newfile->Close();
    f->Close();

    return 0;
}
