#!/bin/bash
export PATH=$PATH:$PWD/python

signals=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
backgrounds=("nonprompt" "diboson" "ttZ")
channels=("Skim1E2Mu" "Skim3Mu" "Combined")

run_saveDataset() {
    local signal="$1"
    local background="$2"
    local channel="$3"
    saveDataset.py --signal $signal --background $background --channel $channel --pilot
}

export -f run_saveDataset

parallel run_saveDataset {1} {2} {3} ::: "${signals[@]}" ::: "${backgrounds[@]}" ::: "${channels[@]}"
