#!/bin/bash

ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("DiMu" "EMu")

for ERA in ${ERAs[@]}; do
    for CHANNEL in ${CHANNELs[@]}; do
        ./scripts/drawPlots.sh $ERA $CHANNEL
    done
done
