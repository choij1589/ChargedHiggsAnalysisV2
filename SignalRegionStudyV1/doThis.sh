#!/bin/bash
# Running combine for all mass points, all channels, all eras
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu")
METHODs=("Baseline" "ParticleNet")
MASSPOINTs=(
    "MHc-70_MA-15" "MHc-70_MA-40" "MHc-70_MA-65"
    "MHc-100_MA-15" "MHc-100_MA-60" "MHc-100_MA-95"
    "MHc-130_MA-15" "MHc-130_MA-55" "MHc-130_MA-90" "MHc-130_MA-125"
    "MHc-160_MA-15" "MHc-160_MA-85" "MHc-160_MA-120" "MHc-160_MA-155"
)
MPForOptimized=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")

combine() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local method=$4
    ./scripts/prepareCombine.sh $era $channel $masspoint $method
    singularity exec --bind /data6/Users/choij/ChargedHiggsAnalysisV2 --bind /cvmfs \
            /data9/Users/choij/Singularity/images/combine ./scripts/runCombine.sh \
            $era $channel $masspoint $method
}

combineChannel() {
    local era=$1
    local masspoint=$2
    local method=$3
    singularity exec --bind /data6/Users/choij/ChargedHiggsAnalysisV2 --bind /cvmfs \
            /data9/Users/choij/Singularity/images/combine ./scripts/runCombineChannel.sh \
            $era Combined $masspoint $method
}

combineEra() {
    local channel=$1
    local masspoint=$2
    local method=$3
    singularity exec --bind /data6/Users/choij/ChargedHiggsAnalysisV2 --bind /cvmfs \
            /data9/Users/choij/Singularity/images/combine ./scripts/runCombineEra.sh \
            FullRun2 $channel $masspoint $method
}

combineAll() {
    local masspoint=$1
    local method=$2
    singularity exec --bind /data6/Users/choij/ChargedHiggsAnalysisV2 --bind /cvmfs \
            /data9/Users/choij/Singularity/images/combine ./scripts/runCombineAll.sh \
            FullRun2 Combined $masspoint $method
}

echo "Cleaning up"
rm -rf samples
rm -rf templates

./scripts/build.sh
for masspoint in ${MASSPOINTs[@]}; do
    for era in ${ERAs[@]}; do
        export -f combine
        parallel -j 2 combine ::: "$era" ::: "${CHANNELs[@]}" ::: "$masspoint" ::: "Baseline"
        wait
    done
done
for masspoint in ${MPForOptimized[@]}; do
    for era in ${ERAs[@]}; do
        export -f combine
        parallel -j 2 combine ::: "$era" ::: "${CHANNELs[@]}" ::: "$masspoint" ::: "ParticleNet"
        wait
    done
done

for masspoint in ${MASSPOINTs[@]}; do
    export -f combineChannel
    parallel -j 2 combine ::: "${ERAs[@]}" ::: "$masspoint" ::: "Baseline"
    wait
done

for masspoint in ${MPForOptimized[@]}; do
    export -f combineChannel
    parallel -j 2 combine ::: "${ERAs[@]}" ::: "$masspoint" ::: "ParticleNet"
    wait
done

for masspoint in ${MASSPOINTs[@]}; do
    export -f combineEra
    parallel -j 2 combine ::: "${CHANNELs[@]}" ::: "$masspoint" ::: "Baseline"
    wait
done

for masspoint in ${MPForOptimized[@]}; do
    export -f combineEra
    parallel -j 2 combine ::: "${CHANNELs[@]}" ::: "$masspoint" ::: "ParticleNet"
    wait
done

for masspoint in ${MASSPOINTs[@]}; do
    export -f combineAll
    combineAll $masspoint "Baseline"
done

for masspoint in ${MPForOptimized[@]}; do
    export -f combineAll
    combineAll $masspoint "ParticleNet"
done
