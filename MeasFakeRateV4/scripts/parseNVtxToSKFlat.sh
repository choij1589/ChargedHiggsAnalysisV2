#!/bin/bash
ERA=$1
export PATH="${PWD}/python:${PATH}"

# Electron
parseNVertexDistribution.py --era $ERA --hlt MeasFakeEl8
parseNVertexDistribution.py --era $ERA --hlt MeasFakeEl12
parseNVertexDistribution.py --era $ERA --hlt MeasFakeEl23
# Muon
parseNVertexDistribution.py --era $ERA --hlt MeasFakeMu8
parseNVertexDistribution.py --era $ERA --hlt MeasFakeMu17

# Copy it to SKFlat
cp results/${ERA}/ROOT/electron/NPV_MeasFakeEl8.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/PileUp
cp results/${ERA}/ROOT/electron/NPV_MeasFakeEl12.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/PileUp
cp results/${ERA}/ROOT/electron/NPV_MeasFakeEl23.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/PileUp
cp results/${ERA}/ROOT/muon/NPV_MeasFakeMu8.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/PileUp
cp results/${ERA}/ROOT/muon/NPV_MeasFakeMu17.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/PileUp
