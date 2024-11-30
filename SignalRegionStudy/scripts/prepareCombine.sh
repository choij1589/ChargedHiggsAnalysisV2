#!/bin/bash
export ERA=$1
export CHANNEL=$2
export MASSPOINT=$3
export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudy/lib:$LD_LIBRARY_PATH

# ParticleNet optimization mass points
masspoints_to_be_optimized=("MHc-100_MA-90" "MHc-130_MA-90" "MHc-160_MA-85")


BASEDIR=$PWD/templates/$ERA/$CHANNEL/$MASSPOINT/Shape/Baseline
if [ -d $BASEDIR ]; then
    echo "WARNING: Directory $BASEDIR already exists"
    rm -rf $BASEDIR
fi

makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT
printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method shape >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/Baseline/datacard.txt

# if $MASSPOINT in masspoints_to_be_optimized
if [[ " ${masspoints_to_be_optimized[@]} " =~ " ${MASSPOINT} " ]]; then
    makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --optimize --update
    printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method ParticleNet >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/ParticleNet/datacard.txt
fi
