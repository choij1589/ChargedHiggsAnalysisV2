#!/bin/bash
export ERA=$1
export CHANNEL=$2
export MASSPOINT=$3
export METHOD=$4
export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudyV1/lib:$LD_LIBRARY_PATH

# ParticleNet optimization mass points
# Replace $CHANNEL, SR1E2Mu -> Skim1E2Mu
SAMPLEDIR=$PWD/samples/$ERA/${CHANNEL/SR/Skim}/$MASSPOINT/$METHOD
BASEDIR=$PWD/templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD
if [ -d $SAMPLEDIR ]; then
    echo "WARNING: Directory $SAMPLEDIR already exists"
    rm -rf $SAMPLEDIR
fi
if [ -d $BASEDIR ]; then
    echo "WARNING: Directory $BASEDIR already exists"
    rm -rf $BASEDIR
fi

preprocess.py --era $ERA --channel ${CHANNEL/SR/Skim} --masspoint $MASSPOINT --method $METHOD
if [ $METHOD == "Baseline" ]; then
    makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
else
    makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD --update
fi
checkTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD \
    >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt
updateEraSuffix.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
