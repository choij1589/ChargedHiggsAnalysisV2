#/bin/bash
source /opt/conda/bin/activate
conda activate pyg
export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"
export PATH="${PATH}:${WORKDIR}/ParticleNet/python"
cd $WORKDIR/ParticleNet
launchHPO.py --signal MHc-130_MA-90 --background nonprompt --channel Combined
