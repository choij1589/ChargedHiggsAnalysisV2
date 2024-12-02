#!/bin/bash
export PATH=$PATH:$PWD/python

signals=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
backgrounds=("nonprompt" "diboson" "ttZ")
channels=("Combined")

trainModel() {
    local signal="$1"
    local background="$2"
    local channel="$3"
    trainModel.py --signal $signal --background $background \
        --channel $channel --model ParticleNet \
        --nNodes 128 --optimizer Adam --initLR 0.002 --scheduler ReduceLROnPlateau \
        --max_epochs 81 --weight_decay 0.001 --fold 0 --device cuda --pilot
}

export -f trainModel
parallel trainModel {1} {2} {3} ::: "${signals[@]}" ::: "${backgrounds[@]}" ::: "${channels[@]}"
