#!/bin/bash
SIGNAL=$1
BACKGROUND=$2
CHANNEL=$3
DEVICE=$4
export PATH=$PATH:$PWD/python

launchGAOptim.py --signal $SIGNAL --background $BACKGROUND --channel $CHANNEL --device $DEVICE
