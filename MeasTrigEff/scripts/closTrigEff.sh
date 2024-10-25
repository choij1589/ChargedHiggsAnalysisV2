#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
PROCESSDILEP=("DYJets" "TTLL_powheg")
PROCESSTRILEP=("ttZToLLNuNu" "WZTo3LNu_amcatnlo" "MHc-70_MA-15" "MHc-100_MA-60" "MHc-130_MA-90" "MHc-160_MA-155")
export PATH=`pwd`/python:$PATH

# use parallel to run the script in parallel
# basic syntax is: closTrigEff.py --era 2016preVFP --channel RunDiMu --process DYJets

parallel closTrigEff.py --era {1} --channel RunDiMu --process {2} ::: ${ERAs[@]} ::: ${PROCESSDILEP[@]}
parallel closTrigEff.py --era {1} --channel RunEMu --process {2} ::: ${ERAs[@]} ::: ${PROCESSDILEP[@]}
parallel closTrigEff.py --era {1} --channel Skim1E2Mu --process {2} ::: ${ERAs[@]} ::: ${PROCESSTRILEP[@]}
parallel closTrigEff.py --era {1} --channel Skim3Mu --process {2} ::: ${ERAs[@]} ::: ${PROCESSTRILEP[@]}
