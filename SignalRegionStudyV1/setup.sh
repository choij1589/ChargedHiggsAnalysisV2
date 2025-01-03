#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export WORKDIR=$PWD/..
cd $WORKDIR/CommonTools/CMSSW_14_1_0_pre4/src
cmsenv
cd -

export PATH=$PWD/python:$PATH
