#!/bin/bash
export PATH=$PATH:$PWD/python

trainModel.py --signal MHc-130_MA-90 --background ttZ \
    --channel Combined --model ParticleNet \
    --nNodes 64 --optimizer RMSprop --initLR 0.001 --scheduler CyclicLR \
    --epochs 80 --device cuda --pilot
