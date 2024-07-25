#!/bin/bash
export PATH=$PATH:$PWD/python

trainModel.py --signal MHc-130_MA-90 --background ttZ \
    --channel Combined --model ParticleNetV2 \
    --nNodes 128 --optimizer RMSprop --initLR 0.0003 --scheduler ExponentialLR \
    --epochs 81 --device cuda 
