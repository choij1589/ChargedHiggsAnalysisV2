#!/bin/bash
ERA=$1

# Copy fake rates to SKFlat
cp results/${ERA}/ROOT/electron/fakerate.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/ID/Electron/fakerate_TopHNT_TopHNL.root
cp results/${ERA}/ROOT/electron/fakerate_qcd.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/ID/Electron/fakerate_qcd_TopHNT_TopHNL.root
cp results/${ERA}/ROOT/muon/fakerate.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/ID/Muon/fakerate_TopHNT_TopHNL.root
cp results/${ERA}/ROOT/muon/fakerate_qcd.root /home/choij/workspace/SKFlatAnalyzer/data/Run2UltraLegacy_v3/${ERA}/ID/Muon/fakerate_qcd_TopHNT_TopHNL.root
