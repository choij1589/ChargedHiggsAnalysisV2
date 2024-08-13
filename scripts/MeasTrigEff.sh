#!/bin/bash
ERA=$1

# This script is used to hadd SKFlatOutput/RunUltraLegacy_v3/MeasTrigEff samples
# NOTE: No need to separate era for leg efficiencies
# NOTE: But should separate period for DblMuTrigEff - mass filter for 2017 B / CDEF

SKFlatOutputDir=$PWD/SKFlatOutput/Run2UltraLegacy_v3

# hadd data
# El legs
cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasElLegs__/DATA
hadd -f MeasTrigEff_SingleMuon.root MeasTrigEff_SingleMuon_*.root
cd -

# Mu legs
if [[ $ERA == '2018' ]]; then
    cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasMuLegs__/DATA
    hadd -f MeasTrigEff_EGamma.root MeasTrigEff_EGamma_*.root
    cd -
else
    cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasMuLegs__/DATA
    hadd -f MeasTrigEff_SingleElectron.root MeasTrigEff_SingleElectron_*.root
    cd -
fi

# EMu DZ
cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasEMuDZ__/DATA
hadd -f MeasTrigEff_MuonEG.root MeasTrigEff_MuonEG_*.root
cd -

# DblMu DZ
if [[ $ERA == '2017' ]]; then
    cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasDblMuDZ__/DATA
    hadd -f MeasTrigEff_DoubleMuon_CDEF.root MeasTrigEff_DoubleMuon_C.root MeasTrigEff_DoubleMuon_D.root MeasTrigEff_DoubleMuon_E.root MeasTrigEff_DoubleMuon_F.root
    cd -
else
    cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasDblMuDZ__/DATA
    hadd -f MeasTrigEff_DoubleMuon.root MeasTrigEff_DoubleMuon_*.root
    cd -
fi

# hadd MC
cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasElLegs__
hadd -f MeasTrigEff_DYJets.root MeasTrigEff_DYJets_*.root
cd -

cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasMuLegs__
hadd -f MeasTrigEff_DYJets.root MeasTrigEff_DYJets_*.root
cd -

cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasEMuDZ__
hadd -f MeasTrigEff_DYJets.root MeasTrigEff_DYJets_*.root
cd -

cd ${SKFlatOutputDir}/MeasTrigEff/$ERA/MeasDblMuDZ__
hadd -f MeasTrigEff_DYJets.root MeasTrigEff_DYJets_*.root
cd -
