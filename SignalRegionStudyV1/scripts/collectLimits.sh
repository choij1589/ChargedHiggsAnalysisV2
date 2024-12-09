#!/bin/bash
export PATH=$PWD/python:$PATH

ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("Combined")
METHODs=("Baseline" "ParticleNet")

collect() {
    local era=$1
    local channel=$2
    local method=$3
    collectLimits.py --era $era --channel $channel --method $method
}

export -f collect
parallel collect {1} {2} {3} ::: "${ERAs[@]}" ::: "${CHANNELs[@]}" ::: "${METHODs[@]}"
