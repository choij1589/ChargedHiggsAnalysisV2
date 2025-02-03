#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "FullRun2")
CHANNELs=("SR1E2Mu" "SR3Mu" "Combined")
METHODs=("Baseline" "ParticleNet")

export PATH=$PWD/python:$PATH
plot_limit() {
    local era=$1
    local channel=$2
    local method=$3
    local limit_type=$4
    plotLimit.py --era $era --channel $channel --method $method --limit_type $limit_type
}

export -f plot_limit
parallel plot_limit ::: ${ERAs[@]} ::: ${CHANNELs[@]} ::: ${METHODs[@]} ::: "Asymptotic"
