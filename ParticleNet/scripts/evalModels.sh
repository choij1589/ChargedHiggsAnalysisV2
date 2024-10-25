#!/bin/bash
SIGNAL=$1
BACKGROUND=$2
CHANNEL=$3
#DEVICE=$4
export PATH=$PATH:$PWD/python

evalModels.py --signal $SIGNAL --background $BACKGROUND --channel $CHANNEL --fold 0 --device cuda:0 &
evalModels.py --signal $SIGNAL --background $BACKGROUND --channel $CHANNEL --fold 1 --device cuda:0 &
evalModels.py --signal $SIGNAL --background $BACKGROUND --channel $CHANNEL --fold 2 --device cuda:1 &
evalModels.py --signal $SIGNAL --background $BACKGROUND --channel $CHANNEL --fold 3 --device cuda:1 &
evalModels.py --signal $SIGNAL --background $BACKGROUND --channel $CHANNEL --fold 4 --device cuda:1 &
