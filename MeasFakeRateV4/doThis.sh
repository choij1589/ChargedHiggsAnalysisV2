#!/bin/bash
ERA=$1
MEASURE=$2

export PATH="${PWD}/python:${PATH}"
rm -rf results/${ERA}

echo "@@@@ parsing integrals..."
parseIntegral.py --era $1 --measure $2
echo "@@@@ measure fake rates..."
measFakeRate.py --era $1 --measure $2
echo "@@@@ plot results..."
plotFakeRate.py --era $1 --measure $2
plotNormalization.py --era $1 --hlt MeasFakeEl12 --wp loose
plotNormalization.py --era $1 --hlt MeasFakeEl12 --wp tight
plotNormalization.py --era $1 --hlt MeasFakeEl23 --wp loose
plotNormalization.py --era $1 --hlt MeasFakeEl23 --wp tight
plotSystematics.py --era $1 --measure $2 --etabin EB1
plotSystematics.py --era $1 --measure $2 --etabin EB2
plotSystematics.py --era $1 --measure $2 --etabin EE
