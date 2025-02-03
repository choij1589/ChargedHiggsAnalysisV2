#!/bin/bash
HOST=`hostname`
echo "@@@@ Working on $HOST"
if [[ $HOST == *"ai-tamsa"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source ~/.conda-activate
    conda activate pyg
elif [[ $HOST == *"lxplus"* ]]; then
	export WORKDIR="/eos/user/c/choij/ChargedHiggsAnalysisV2"
	source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
elif [[ $HOST == *"knu"* ]]; then
    export WORKDIR="/u/user/choij/scratch/ChargedHiggsAnalysisV2"
    export SKFlatOutput="/u/user/choij/SE_USERHOME/SKFlatOutput/Run2UltraLegacy_v3"
    source /u/user/choij/miniconda3/bin/activate
    conda activate pyg
    unset LD_PRELOAD
elif [[ $HOST == *"tamsa"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-centos7-gcc12-opt/setup.sh
elif [[ $HOST == *"cms"* ]]; then
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
    source /opt/conda/bin/activate
    conda activate pyg
elif [[ $HOST == *"private"* ]]; then
    export WORKDIR="$HOME/workspace/ChargedHiggsAnalysisV2"
    source ~/.conda-activate
    conda activate pyg
elif [[ $HOST == *"Mac"* ]]; then
    export WORKDIR="$PWD"
    source $WORKDIR/myenv/bin/activate
    source root_install/bin/thisroot.sh
else
    echo "Unknown host"
    return 1
fi

export PYTHONPATH=$WORKDIR/CommonTools/python:$PYTHONPATH
export PYTHONPATH=$WORKDIR/CommonTools/tdr-style:$PYTHONPATH
echo "@@@@ WORKDIR=$WORKDIR"

# build 
cd $WORKDIR/ParticleNet/libs
g++ `root-config --cflags --libs` -o copytree copytree.cc
cd -
