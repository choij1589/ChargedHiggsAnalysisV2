#!/bin/bash
export PATH=$PATH:$PWD/python

signals=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
backgrounds=("nonprompt" "diboson" "ttZ")

for signal in "${signals[@]}"; do
    for background in "${backgrounds[@]}"; do
        saveDataset.py --signal "$signal" --background "$background" --channel Combined
    done
done
