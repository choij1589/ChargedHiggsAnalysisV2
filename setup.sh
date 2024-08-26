#!/bin/bash
HOST=`hostname`
echo "@@@@ Working on $HOST"
if [[ $HOST == *"ai-tamsa"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source ~/.conda-activate
    conda activate pyg
elif [[ $HOST == *"tamsa"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source /opt/conda/bin/activate
    conda activate pyg
elif [[ $HOST == *"cms"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source /opt/conda/bin/activate
    conda activate pyg
elif [[ $HOST == *"Mac"* ]]; then
    export WORKDIR="$HOME/workspace/ChargedHiggsAnalysisV2"
    source ~/.conda-activate
    conda activate pyg
else
    echo "Unknown host"
    return 1
fi

export PYTHONPATH=$WORKDIR/CommonTools/python:$PYTHONPATH
export PYTHONPATH=$WORKDIR/CommonTools/tdr-style:$PYTHONPATH
echo "@@@@ WORKDIR=$WORKDIR"

git status
