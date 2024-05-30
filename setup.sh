#!/bin/bash
echo "@@@@ Working on `hostname`"
if [[ `hostname` == *"tamsa"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"

    echo "Need additional ROOT and python configuration based on LCG"
elif [[ `hostname` == *"cms"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source $HOME/miniconda3/bin/activate
    conda activate pyg
else
    export WORKDIR="$HOME/workspace/ChargedHiggsAnalysisV2"
    source ~/.conda-activate
    conda activate pyg
fi
echo "@@@@ WORKDIR=$WORKDIR"
