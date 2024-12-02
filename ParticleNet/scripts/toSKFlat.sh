#!/bin/bash
CHANNEL=$1

if [ $HOSTNAME == "private-snu" ]; then
    SKFlatDataPath="/home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/"
elif [ $HOSTNAME == "ai-tamsa"* ]; then
    SKFlatDataPath="/data6/Users/choij/SKFlatAnalyzer/data/Run2UltraLegacy_v3/"
else
    echo "Unknown machine"
    exit
fi

mkdir -p $SKFlatDataPath/Classifiers/ParticleNet
if [ -d $SKFlatDataPath/Classifiers/ParticleNet/${CHANNEL}__ ]; then
    echo "Directory already exists, backup..."
    mv $SKFlatDataPath/Classifiers/ParticleNet/${CHANNEL}__ $SKFlatDataPath/Classifiers/ParticleNet/${CHANNEL}__backup_$(date +%Y%m%d)
fi
cp -r results/${CHANNEL}__ $SKFlatDataPath/Classifiers/ParticleNet
