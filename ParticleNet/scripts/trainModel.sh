#!/bin/bash
export PATH=$PATH:$PWD/python

trainModel.py --signal MHc-130_MA-90 --background ttZ \
    --channel Combined --model ParticleNet \
    --nNodes 128 --optimizer Adam --initLR 0.002 --scheduler ReduceLROnPlateau \
    --max_epochs 81 --weight_decay 0.001 --device cuda
