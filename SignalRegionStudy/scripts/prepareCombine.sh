#!/bin/bash
export ERA=$1
export CHANNEL=$2
export MASSPOINT=$3
export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudy/lib:$LD_LIBRARY_PATH

BASEDIR=$PWD/templates/$ERA/$CHANNEL/MHc-130_MA-90/Shape/Baseline
if [ -d $BASEDIR ]; then
    echo "WARNING: Directory $BASEDIR already exists"
    rm -rf $BASEDIR
fi

makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT
printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method shape >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/Baseline/datacard.txt
