#!/bin/bash
ERA=$1

./scripts/build.sh
./scripts/fitMT.sh $ERA MeasFakeEl12
./scripts/fitMT.sh $ERA MeasFakeEl23
./scripts/parseIntegral.sh $ERA
