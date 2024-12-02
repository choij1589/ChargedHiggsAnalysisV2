#!/bin/bash
SIGNALs=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
BACKGROUNDs=("nonprompt" "diboson" "ttZ")
CHANNEL="Combined"

SKFlatParticleNetPath="/data6/Users/choij/SKFlatAnalyzer/data/Run2UltraLegacy_v3/Classifiers/ParticleNet/${CHANNEL}__"
for sig in ${SIGNALs[@]}; do
    for bkg in ${BACKGROUNDs[@]}; do
        mkdir -p ${SKFlatParticleNetPath}/${sig}_vs_${bkg}/pilot
        cp results/${CHANNEL}__/${sig}_vs_${bkg}/pilot/models/ParticleNet-nNodes128-Adam-initLR0p0020-decay0p00100-ReduceLROnPlateau.pt \
            ${SKFlatParticleNetPath}/${sig}_vs_${bkg}/pilot/ParticleNet.pt
        echo "MHc-130_MA-90, nonprompt, 1, 128, Adam, 0.002, ReduceLROnPlateau, 0.001, 0.8813043456186247, 0.8714873219363756, 0.877795841375898, 0.0, 0.012" \
            >> ${SKFlatParticleNetPath}/${sig}_vs_${bkg}/pilot/summary.txt
    done
done
