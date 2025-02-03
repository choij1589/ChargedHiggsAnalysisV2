#!/bin/bash
ERA=$1
HLT=$2

export PATH="${PWD}/python:${PATH}"

fitMT.py --era $ERA --hlt $HLT --wp loose --syst Central
fitMT.py --era $ERA --hlt $HLT --wp loose --syst RequireHeavyTag
fitMT.py --era $ERA --hlt $HLT --wp tight --syst Central
fitMT.py --era $ERA --hlt $HLT --wp tight --syst RequireHeavyTag
