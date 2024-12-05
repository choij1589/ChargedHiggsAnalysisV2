#!/bin/bash
export ERA=$1
export CHANNEL=$2
export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudy/lib:$LD_LIBRARY_PATH


SIGNALs=(
    "MHc-70_MA-15" "MHc-70_MA-40" "MHc-70_MA-65"
    "MHc-100_MA-15" "MHc-100_MA-60" "MHc-100_MA-95"
    "MHc-130_MA-15" "MHc-130_MA-55" "MHc-130_MA-90" "MHc-130_MA-125"
    "MHc-160_MA-15" "MHc-160_MA-85" "MHc-160_MA-120" "MHc-160_MA-155"
)

preprocess() {
    local signal=$1
    preprocess.py --era $ERA --channel $CHANNEL --signal $signal
}

export -f preprocess
# Too hadd data
preprocess.py --era $ERA --channel $CHANNEL --signal "MHc-70_MA-15"
parallel -j 14 preprocess ::: ${SIGNALs[@]}
