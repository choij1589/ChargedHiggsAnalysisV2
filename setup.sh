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
    ### Warning message with red color
    if [[ `hostname` == *"Mac"* ]]; then
        echo -e "\e[31m@@@@ WARNING: pyg usage deprecated in MacOS\e[0m"
    fi
fi

export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pyg/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export PATH=$HOME/miniconda3/envs/pyg/lib/python3.11/site-packages/nvidia/cuda_runtime/bin:$PATH

export PYTHONPATH=$WORKDIR/CommonTools/python:$PYTHONPATH
export PYTHONPATH=$WORKDIR/CommonTools/tdr-style:$PYTHONPATH
echo "@@@@ WORKDIR=$WORKDIR"
