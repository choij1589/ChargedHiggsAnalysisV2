#!/bin/bash
ERA=$1
HLT=$2

python python/fitMT.py --era $ERA --hlt $HLT --wp loose --syst Central
python python/fitMT.py --era $ERA --hlt $HLT --wp loose --syst RequireHeavyTag
python python/fitMT.py --era $ERA --hlt $HLT --wp tight --syst Central
python python/fitMT.py --era $ERA --hlt $HLT --wp tight --syst RequireHeavyTag
