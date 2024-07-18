#!/bin/bash

ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
for ERA in "${ERAs[@]}"
do
    ./scripts/measFakeRate.sh $ERA muon
    ./scripts/measFakeRate.sh $ERA electron
done

./scripts/plotSystematics.sh
