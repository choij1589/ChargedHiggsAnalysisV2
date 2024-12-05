#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu")
MASSPOINTs=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
SKFlatDataPath="/data6/Users/choij/SKFlatAnalyzer/data/Run2UltraLegacy_v3/"

for ERA in ${ERAs[@]}; do
    for CHANNEL in ${CHANNELs[@]}; do
        for MASSPOINT in ${MASSPOINTs[@]}; do
            mkdir -p $SKFlatDataPath/$ERA/classifiers/$CHANNEL/$MASSPOINT
            cp templates/$ERA/$CHANNEL/$MASSPOINT/Shape/ParticleNet/models.pkl $SKFlatDataPath/$ERA/classifiers/$CHANNEL/$MASSPOINT/models.pkl
        done
    done
done
