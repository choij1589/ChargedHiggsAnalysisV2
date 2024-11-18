#!/bin/bash
CHANNEL=$1

echo Currently parsing done in GAMSA only - for other machines, please modify the path
export SKFlatDataPath="/data6/Users/choij/SKFlatAnalyzer/data/Run2UltraLegacy_v3/"
mkdir -p $SKFlatDataPath/Classifiers/ParticleNet
if [ -d $SKFlatDataPath/Classifiers/ParticleNet/$CHANNEL ]; then
    echo "Removing existing directory"
    rm -rf $SKFlatDataPath/Classifiers/ParticleNet/$CHANNEL
fi
cp -r results/$CHANNL $SKFlatDataPath/Classifiers/ParticleNet
