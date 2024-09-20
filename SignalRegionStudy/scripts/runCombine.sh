#!/bin/bash
ERA=$1
CHANNEL=$2
MASSPOINT=$3

echo $PWD
docker run --rm --user cmsusr \
    -v /cvmfs:/cvmfs:shared \
    -v ~/workspace:/home/cmsusr/workspace \
    choij1589/alma8:latest \
    bash /home/cmsusr/workspace/ChargedHiggsAnalysisV2/SignalRegionStudy/scripts/docker_command.sh $ERA $CHANNEL $MASSPOINT
