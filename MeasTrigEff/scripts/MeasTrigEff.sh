#!/bin/bash
ERA=$1

export PATH="$WORKDIR/MeasTrigEff/python":$PATH
measEMuLegEff.py --era $ERA --hltpath Mu8El23 --leg muon
measEMuLegEff.py --era $ERA --hltpath Mu8El23 --leg electron
measEMuLegEff.py --era $ERA --hltpath Mu23El12 --leg muon
measEMuLegEff.py --era $ERA --hltpath Mu23El12 --leg electron

if [[ $ERA == "2016postVFP" ]]; then
    measPairwiseFilterEff.py --era $ERA --filter EMuDZ >> results/$ERA/PairwiseFilterEff_EMuDZ.csv
    measPairwiseFilterEff.py --era $ERA --filter DblMuDZ >> results/$ERA/PairwiseFilterEff_DblMuDZ.csv
elif [[ $ERA == "2017" ]]; then
    measPairwiseFilterEff.py --era $ERA --filter EMuDZ >> results/$ERA/PairwiseFilterEff_EMuDZ.csv
    measPairwiseFilterEff.py --era $ERA --filter DblMuDZ >> results/$ERA/PairwiseFilterEff_DblMuDZ.csv
    measPairwiseFilterEff.py --era $ERA --filter DblMuDZM >> results/$ERA/PairwiseFilterEff_DblMuDZM.csv
    measPairwiseFilterEff.py --era $ERA --filter DblMuM >> results/$ERA/PairwiseFilterEff_DblMuM.csv
elif [[ $ERA == "2018" ]]; then
    measPairwiseFilterEff.py --era $ERA --filter EMuDZ >> results/$ERA/PairwiseFilterEff_EMuDZ.csv
    measPairwiseFilterEff.py --era $ERA --filter DblMuDZM >> results/$ERA/PairwiseFilterEff_DblMuDZM.csv
    measPairwiseFilterEff.py --era $ERA --filter DblMuM >> results/$ERA/PairwiseFilterEff_DblMuM.csv
else
    echo "No pairwise filter efficiency measurement for $ERA."
fi
