#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu")
MASSPOINTs=("MHc-70_MA-15" "MHc-70_MA-40" "MHc-70_MA-65"
            "MHc-100_MA-15" "MHc-100_MA-60" "MHc-100_MA-95"
            "MHc-130_MA-15" "MHc-130_MA-55" "MHc-130_MA-90" "MHc-130_MA-125"
            "MHc-160_MA-15" "MHc-160_MA-85" "MHc-160_MA-120" "MHc-160_MA-155")
MASSPOINTs_OPTIM=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
export PATH=$PWD/python:$PATH

plot() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local method=$4
    plotTemplates.py --era $era --channel $channel --masspoint $masspoint --method $method
}
export -f plot
parallel plot ::: ${ERAs[@]} ::: ${CHANNELs[@]} ::: ${MASSPOINTs[@]} ::: "Baseline"
parallel plot ::: ${ERAs[@]} ::: ${CHANNELs[@]} ::: ${MASSPOINTs_OPTIM[@]} ::: "ParticleNet"
