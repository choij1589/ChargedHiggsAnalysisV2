#!/bin/bash
ERA=$1
MEASURE=$2

export PATH="${PWD}/python:${PATH}"

echo "@@@@ parsing integrals..."
parseIntegral.py --era $1 --measure $2
echo "@@@@ measure fake rates..."
measFakeRate.py --era $1 --measure $2
measFakeRate.py --era $1 --measure $2 --isQCD
echo "@@@@ plot results..."
plotFakeRate.py --era $1 --measure $2
plotFakeRate.py --era $1 --measure $2 --isQCD

if [ $MEASURE = "electron" ]; then
    plotNormalization.py --era $1 --hlt MeasFakeEl12 --wp loose
    plotNormalization.py --era $1 --hlt MeasFakeEl12 --wp tight
    plotNormalization.py --era $1 --hlt MeasFakeEl23 --wp loose
    plotNormalization.py --era $1 --hlt MeasFakeEl23 --wp tight
    plotValidation.py --era $1 --hlt MeasFakeEl12 --wp loose --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeEl12 --wp tight --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeEl23 --wp loose --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeEl23 --wp tight --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeEl12 --wp loose --region Inclusive --syst RequireHeavyTag
    plotValidation.py --era $1 --hlt MeasFakeEl12 --wp tight --region Inclusive --syst RequireHeavyTag
    plotValidation.py --era $1 --hlt MeasFakeEl23 --wp loose --region Inclusive --syst RequireHeavyTag
    plotValidation.py --era $1 --hlt MeasFakeEl23 --wp tight --region Inclusive --syst RequireHeavyTag
elif [ $MEASURE = "muon" ]; then
    plotNormalization.py --era $1 --hlt MeasFakeMu8 --wp loose
    plotNormalization.py --era $1 --hlt MeasFakeMu8 --wp tight
    plotNormalization.py --era $1 --hlt MeasFakeMu17 --wp loose
    plotNormalization.py --era $1 --hlt MeasFakeMu17 --wp tight
    plotValidation.py --era $1 --hlt MeasFakeMu8 --wp loose --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeMu8 --wp tight --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeMu17 --wp loose --region Inclusive
    plotValidation.py --era $1 --hlt MeasFakeMu17 --wp tight --region Inclusive
else
    echo "ERROR: measure must be either electron or muon"
    exit 1
fi
