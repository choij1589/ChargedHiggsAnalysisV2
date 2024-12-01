#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export WORKDIR=/home/cmsusr/workspace/ChargedHiggsAnalysisV2
#export WORKDIR=/home/choij/ChargedHiggsAnalysisV2
cd $WORKDIR/CommonTools/CMSSW_14_1_0_pre4/src
cmsenv
cd -

export ROOT_DIR=/cvmfs/cms.cern.ch/el8_amd64_gcc12/lcg/root/6.30.07-f3322c77db1c59847b28fde88ff7218c/cmake
