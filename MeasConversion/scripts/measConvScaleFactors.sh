#!/bin/bash

export PATH="${PWD}/python:${PATH}"
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("Skim1E2Mu" "Skim3Mu")

# Loop over ERAs and CHANNELs using GNU parallel
# measConvScaleFactor.py --era $ERA --channel $CHANNEL
parallel -j 8 -k "measConvScaleFactor.py --era {1} --channel {2}" ::: ${ERAs[@]} ::: ${CHANNELs[@]}

