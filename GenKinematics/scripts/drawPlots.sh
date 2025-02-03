#!/bin/bash
ERA=$1
CHANNEL=$2

export PATH="${PWD}/python:${PATH}"
export ERA
export CHANNEL

draw_plot() {
    local histkey=$1
    plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey"
    plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --mHc 70
    plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --mHc 100
    plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --mHc 130
    plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --mHc 160
}

if [ $CHANNEL == "GEN1E2Mu" ]; then
    histkeys=(
        "muons_fromA/1/pt" "muons_fromA/1/eta" "muons_fromA/1/phi"
        "muons_fromA/2/pt" "muons_fromA/2/eta" "muons_fromA/2/phi"
        "electrons_fromEW/pt" "electrons_fromEW/eta" "electrons_fromEW/phi"
        "electrons_fromOffshellW/pt" "electrons_fromOffshellW/eta" "electrons_fromOffshellW/phi"
        "b_withHc/pt" "b_withHc/eta" "b_withHc/phi"
        "b_withW/pt" "b_withW/eta" "b_withW/phi"
    )
    export -f draw_plot
    parallel draw_plot ::: "${histkeys[@]}"
elif [ $CHANNEL == "GEN3Mu" ]; then
    histkeys=(
        "muons_fromA/1/pt" "muons_fromA/1/eta" "muons_fromA/1/phi"
        "muons_fromA/2/pt" "muons_fromA/2/eta" "muons_fromA/2/phi"
        "muons_fromEW/pt" "muons_fromEW/eta" "muons_fromEW/phi"
        "muons_fromOffshellW/pt" "muons_fromOffshellW/eta" "muons_fromOffshellW/phi"
        "b_withHc/pt" "b_withHc/eta" "b_withHc/phi"
        "b_withW/pt" "b_withW/eta" "b_withW/phi"
    )
    export -f draw_plot
    parallel draw_plot ::: "${histkeys[@]}"
fi
