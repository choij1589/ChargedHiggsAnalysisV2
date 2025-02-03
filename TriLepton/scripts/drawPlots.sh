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
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi" "electrons/1/charge" "electrons/1/px" "electrons/1/py" "electrons/1/pz" "electrons/1/energy"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nonprompt/pt" "nonprompt/eta"
        "MHc-100_MA-95/score_nonprompt" "MHc-100_MA-95/score_diboson" "MHc-100_MA-95/score_ttZ" "MHc-100_MA-95/score"
        "MHc-130_MA-90/score_nonprompt" "MHc-130_MA-90/score_diboson" "MHc-130_MA-90/score_ttZ" "MHc-130_MA-90/score"
        "MHc-160_MA-85/score_nonprompt" "MHc-160_MA-85/score_diboson" "MHc-160_MA-85/score_ttZ" "MHc-160_MA-85/score"
 
    )
    plot.py --era $ERA --channel $CHANNEL --histkey "muons/1/pt"
    export -f draw_plot
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZFake3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy"
        "muons/3/pt" "muons/3/eta" "muons/3/phi" "muons/3/charge" "muons/3/px" "muons/3/py" "muons/3/pz" "muons/3/energy"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nZCand/pt" "nZCand/eta" "nZCand/phi" "nZCand/mass"
        "nonprompt/pt" "nonprompt/eta"
        "MHc-100_MA-95/score_nonprompt" "MHc-100_MA-95/score_diboson" "MHc-100_MA-95/score_ttZ" "MHc-100_MA-95/score"
        "MHc-130_MA-90/score_nonprompt" "MHc-130_MA-90/score_diboson" "MHc-130_MA-90/score_ttZ" "MHc-130_MA-90/score"
        "MHc-160_MA-85/score_nonprompt" "MHc-160_MA-85/score_diboson" "MHc-160_MA-85/score_ttZ" "MHc-160_MA-85/score"

    )
    plot.py --era $ERA --channel $CHANNEL --histkey "muons/1/pt"
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
    plot.py --era $ERA --channel $CHANNEL --histkey "muons/1/pt"
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
    plot.py --era $ERA --channel $CHANNEL --histkey "muons/1/pt"
    export -f draw_plot
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "SR1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi" "electrons/1/charge" "electrons/1/px" "electrons/1/py" "electrons/1/pz" "electrons/1/energy"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge" "bjets/1/btagScore"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge" "bjets/2/btagScore"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge" "bjets/3/btagScore"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "pair/pt" "pair/eta" "pair/phi" "pair/mass"
        "MHc-100_MA-95/score_nonprompt" "MHc-100_MA-95/score_diboson" "MHc-100_MA-95/score_ttZ" "MHc-100_MA-95/score"
        "MHc-130_MA-90/score_nonprompt" "MHc-130_MA-90/score_diboson" "MHc-130_MA-90/score_ttZ" "MHc-130_MA-90/score"
        "MHc-160_MA-85/score_nonprompt" "MHc-160_MA-85/score_diboson" "MHc-160_MA-85/score_ttZ" "MHc-160_MA-85/score"
    )
    plot.py --era $ERA --channel $CHANNEL --histkey "muons/1/pt" --blind
    export -f draw_plot_blind
    parallel draw_plot_blind ::: ${histkeys[@]}
elif [[ $CHANNEL == "SR3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy"
        "muons/3/pt" "muons/3/eta" "muons/3/phi" "muons/3/charge" "muons/3/px" "muons/3/py" "muons/3/pz" "muons/3/energy"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/btagScore" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/btagScore" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/btagScore" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/btagScore" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge" "bjets/1/btagScore"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge" "bjets/2/btagScore"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge" "bjets/3/btagScore"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "stack/pt" "stack/eta" "stack/phi" "stack/mass"
        "MHc-100_MA-95/score_nonprompt" "MHc-100_MA-95/score_diboson" "MHc-100_MA-95/score_ttZ" "MHc-100_MA-95/score"
        "MHc-130_MA-90/score_nonprompt" "MHc-130_MA-90/score_diboson" "MHc-130_MA-90/score_ttZ" "MHc-130_MA-90/score"
        "MHc-160_MA-85/score_nonprompt" "MHc-160_MA-85/score_diboson" "MHc-160_MA-85/score_ttZ" "MHc-160_MA-85/score"
    )
    plot.py --era $ERA --channel $CHANNEL --histkey "muons/1/pt" --blind
    export -f draw_plot_blind
    parallel draw_plot_blind ::: ${histkeys[@]}
else
    echo "Channel $CHANNEL not implemented yet"
fi
