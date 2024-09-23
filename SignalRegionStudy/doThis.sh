#!/bin/bash
# Running combine for all mass points, all channels, all eras
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu")
MASSPOINTs=(
    "MHc-70_MA-15" "MHc-70_MA-40" "MHc-70_MA-65"
    "MHc-100_MA-15" "MHc-100_MA-60" "MHc-100_MA-95"
    "MHc-130_MA-15" "MHc-130_MA-55" "MHc-130_MA-90" "MHc-130_MA-125"
    "MHc-160_MA-15" "MHc-160_MA-85" "MHc-160_MA-120" "MHc-160_MA-155"
)

combine() {
    local era=$1
    local channel=$2
    local masspoint=$3
    ./scripts/prepareCombine.sh $era $channel $masspoint
    ./scripts/runCombine.sh $era $channel $masspoint
}

for era in ${ERAs[@]}; do
  for channel in ${CHANNELs[@]}; do
    ./scripts/preprocess.sh $era ${channel/SR/Skim}
    export -f combine
    parallel -j 14 combine ::: "$era" ::: "$channel" ::: "${MASSPOINTs[@]}"
  done
done
