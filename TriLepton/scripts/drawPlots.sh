#!/bin/bash
ERA=$1
CHANNEL=$2
export PATH="${PWD}/python:${PATH}"
export ERA
export CHANNEL

draw_plot() {
    local histkey=$1
    plot.py --era $ERA --channel $CHANNEL --histkey $histkey
}
draw_plot_blind() {
    local histkey=$1
    plot.py --era $ERA --channel $CHANNEL --histkey $histkey --blind
}

if [[ $CHANNEL == "ZFake1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore"
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nonprompt/pt" "nonprompt/eta"
    )
    export -f draw_plot
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZFake3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "muons/3/pt" "muons/3/eta" "muons/3/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore"
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nZCand/pt" "nZCand/eta" "nZCand/phi" "nZCand/mass"
        "nonprompt/pt" "nonprompt/eta"
    )
    export -f draw_plot
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZGamma1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi"
        "jets/size" "bjets/size" 
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" 
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" 
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" 
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" 
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" 
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "convLep/pt" "convLep/eta" "convLep/phi"
    )
    export -f draw_plot
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZGamma3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "muons/3/pt" "muons/3/eta" "muons/3/phi"
        "jets/size" 
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" 
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" 
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" 
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" 
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" 
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "convLep/pt" "convLep/eta" "convLep/phi"
    )
    export -f draw_plot
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "SR1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge" "bjets/1/btagScore"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge" "bjets/2/btagScore"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge" "bjets/3/btagScore"
        "METv/pt" "METv/phi"
        "pair/pt" "pair/eta" "pair/phi" "pair/mass"
        "MHc-100_MA-95/score_nonprompt" "MHc-100_MA-95/score_diboson" "MHc-100_MA-95/score_ttZ"
        "MHc-130_MA-90/score_nonprompt" "MHc-130_MA-90/score_diboson" "MHc-130_MA-90/score_ttZ"
        "MHc-160_MA-85/score_nonprompt" "MHc-160_MA-85/score_diboson" "MHc-160_MA-85/score_ttZ"
    )
    export -f draw_plot_blind
    parallel draw_plot_blind ::: ${histkeys[@]}
elif [[ $CHANNEL == "SR3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "muons/3/pt" "muons/3/eta" "muons/3/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge" "bjets/1/btagScore"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge" "bjets/2/btagScore"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge" "bjets/3/btagScore"
        "METv/pt" "METv/phi"
        "stack/pt" "stack/eta" "stack/phi" "stack/mass"
        "MHc-100_MA-95/score_nonprompt" "MHc-100_MA-95/score_diboson" "MHc-100_MA-95/score_ttZ"
        "MHc-130_MA-90/score_nonprompt" "MHc-130_MA-90/score_diboson" "MHc-130_MA-90/score_ttZ"
        "MHc-160_MA-85/score_nonprompt" "MHc-160_MA-85/score_diboson" "MHc-160_MA-85/score_ttZ"
    )
    export -f draw_plot_blind
    parallel draw_plot_blind ::: ${histkeys[@]}
else
    echo "Channel $CHANNEL not implemented yet"
fi
