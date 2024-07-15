#!/bin/bash
ERA=$1
CHANNEL=$2

if [ $CHANNEL = "Skim1E2Mu" ]; then
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/1/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/1/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/1/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/2/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/2/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/2/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key electrons/1/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key electrons/1/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key electrons/1/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/1/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/1/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/1/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/2/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/2/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/2/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/3/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/3/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/3/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/4/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/4/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/4/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/size
    python drawPlots.py --era $ERA --channel $CHANNEL --key METv/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key METv/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/mass
    python drawPlots.py --era $ERA --channel $CHANNEL --key nonprompt/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key nonprompt/eta
elif [ $CHANNEL = "Skim3Mu" ]; then
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/1/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/1/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/1/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/2/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/2/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/2/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/3/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/3/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key muons/3/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/1/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/1/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/1/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/2/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/2/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/2/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/3/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/3/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/3/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/4/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/4/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/4/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key jets/size
    python drawPlots.py --era $ERA --channel $CHANNEL --key METv/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key METv/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key ZCand/mass
    python drawPlots.py --era $ERA --channel $CHANNEL --key nZCand/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key nZCand/eta
    python drawPlots.py --era $ERA --channel $CHANNEL --key nZCand/phi
    python drawPlots.py --era $ERA --channel $CHANNEL --key nZCand/mass
    python drawPlots.py --era $ERA --channel $CHANNEL --key nonprompt/pt
    python drawPlots.py --era $ERA --channel $CHANNEL --key nonprompt/eta
else
    echo "Invalid channel"
fi
