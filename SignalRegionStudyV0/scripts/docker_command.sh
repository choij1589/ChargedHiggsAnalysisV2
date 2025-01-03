#!/bin/bash
ERA=$1
CHANNEL=$2
MASSPOINT=$3
source /cvmfs/cms.cern.ch/cmsset_default.sh
export WORKDIR=/home/cmsusr/workspace/ChargedHiggsAnalysisV2
cd $WORKDIR/CommonTools/CMSSW_14_1_0_pre4/src
eval `scramv1 runtime -sh`

# Move to template directory
cd $WORKDIR/SignalRegionStudy/templates/$ERA/$CHANNEL/$MASSPOINT/Shape/Baseline

# Run the command
text2workspace.py datacard.txt -o workspace.root
combine -M FitDiagnostics workspace.root
combine -M AsymptoticLimits workspace.root -t -1
combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.025 # 95% Down
combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.160 # 68% Down
combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.500 # Expected
combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.840 # 68% Up
combine -M HybridNew --LHCmode LHC-limits workspace.root --saveHybridResult -t -1 --expectedFromGrid 0.975 # 95% Up

combineTool.py -M Impacts -d workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --setParameterRanges r=-1,1
combineTool.py -M Impacts -d workspace.root -m 125 --robustFit 1 -t -1 --doFits --setParameterRanges r=-1,1
combineTool.py -M Impacts -d workspace.root -m 125 -o impacts.json
plotImpacts.py -i impacts.json -o impacts
