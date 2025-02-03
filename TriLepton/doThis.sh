#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu" "ZFake1E2Mu" "ZFake3Mu" "ZGamma1E2Mu" "ZGamma3Mu")

plotAll() {
    local era=$1
    local channel=$2
    ./scripts/drawPlots.sh $era $channel
}

export -f plotAll
parallel -j 4 plotAll ::: "${ERAs[@]}" ::: "${CHANNELs[@]}"


