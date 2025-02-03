#!/bin/bash
export ERA=$1                  # 2016preVFP 2016postVFP 2017 2018 FullRun2
export CHANNEL=$2              # SR1E2Mu SR3Mu Combined
export MASSPOINT=$3
export METHOD=$4

source /cvmfs/cms.cern.ch/cmsset_default.sh
export WORKDIR=/data6/Users/choij/ChargedHiggsAnalysisV2
cd $WORKDIR/CommonTools/CMSSW_14_1_0_pre4/src
eval `scramv1 runtime -sh`

export BASEDIR=$WORKDIR/SignalRegionStudyV1/templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD
# Combine channel first

if [ $CHANNEL == "Combined" ]; then
    if [ -d $BASEDIR ]; then rm -rf $BASEDIR; fi
    mkdir -p $BASEDIR && cd $BASEDIR
    combineCards.py \
        ch1e2mu=$WORKDIR/SignalRegionStudyV1/templates/$ERA/SR1E2Mu/$MASSPOINT/Shape/$METHOD/datacard.txt \
        ch3mu=$WORKDIR/SignalRegionStudyV1/templates/$ERA/SR3Mu/$MASSPOINT/Shape/$METHOD/datacard.txt >> datacard.txt
    cd -
fi

if [ $ERA == "FullRun2" ]; then
    if [ -d $BASEDIR ]; then rm -rf $BASEDIR; fi
    mkdir -p $BASEDIR && cd $BASEDIR
    combineCards.py \
        era16a=$WORKDIR/SignalRegionStudyV1/templates/2016preVFP/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
        era16b=$WORKDIR/SignalRegionStudyV1/templates/2016postVFP/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
        era17=$WORKDIR/SignalRegionStudyV1/templates/2017/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
        era18=$WORKDIR/SignalRegionStudyV1/templates/2018/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt >> datacard.txt
    cd -
fi

cd $BASEDIR

# Run the combine
text2workspace.py datacard.txt -o workspace.root
combine -M FitDiagnostics workspace.root
combine -M AsymptoticLimits workspace.root -t -1
#combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.025 & # 95% Down
#combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.160 & # 68% Down
#combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.500 & # Expected
#combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.840 & # 68% Up
#combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.975 & # 95% Up
#wait
combineTool.py -M Impacts -d workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --setParameterRanges r=-1,1
combineTool.py -M Impacts -d workspace.root -m 125 --robustFit 1 -t -1 --doFits --setParameterRanges r=-1,1
combineTool.py -M Impacts -d workspace.root -m 125 -o impacts.json
plotImpacts.py -i impacts.json -o impacts
