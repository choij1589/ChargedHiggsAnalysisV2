#!/bin/bash
ERA=$1

export PATH="$WORKDIR/MeasTrigEff/python":$PATH
measEMuLegEff.py --era $ERA --hltpath Mu8El23 --leg muon
measEMuLegEff.py --era $ERA --hltpath Mu8El23 --leg electron
measEMuLegEff.py --era $ERA --hltpath Mu23El12 --leg muon
measEMuLegEff.py --era $ERA --hltpath Mu23El12 --leg electron
