#!/bin/bash
echo "@@@@ Working on $HOSTNAME"
if [[ $HOSTNAME == *"tamsa"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source /opt/conda/bin/activate
    conda activate pyg
elif [[ $HOSTNAME == *"cms"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source /opt/conda/bin/activate
    conda activate pyg
else
    export WORKDIR="$HOME/workspace/ChargedHiggsAnalysisV2"
    source ~/.conda-activate
    conda activate pyg
    ### Warning message with red color
    if [[ `hostname` == *"Mac"* ]]; then
        echo -e "\e[31m@@@@ WARNING: pyg usage deprecated in MacOS\e[0m"
    fi
fi

export PYTHONPATH=$WORKDIR/CommonTools/python:$PYTHONPATH
export PYTHONPATH=$WORKDIR/CommonTools/tdr-style:$PYTHONPATH
echo "@@@@ WORKDIR=$WORKDIR"
