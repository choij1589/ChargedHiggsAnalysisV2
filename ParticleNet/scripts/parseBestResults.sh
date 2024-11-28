#!/bin/bash
SIGNALs=("MHc-100_MA-95" "MHc-130_MA-90" "MHc-160_MA-85")
BACKGROUNDs=("nonprompt" "diboson" "ttZ")
CHANNELs=("Combined")
export PATH=$PWD/python:$PATH

parse_result() {
    local signal=$1
    local background=$2
    local channel=$3
    parseBestResults.py --signal $signal --background $background --channel $channel
}

export -f parse_result
parallel parse_result ::: ${SIGNALs[@]} ::: ${BACKGROUNDs[@]} ::: ${CHANNELs[@]}
