#!/bin/bash
export PATH="${PWD}/python:${PATH}"
plotIDVariables.py --era 2016preVFP --region EB1 &
plotIDVariables.py --era 2016preVFP --region EB2 &
plotIDVariables.py --era 2016preVFP --region EE &

plotIDVariables.py --era 2016postVFP --region EB1 &
plotIDVariables.py --era 2016postVFP --region EB2 &
plotIDVariables.py --era 2016postVFP --region EE &

plotIDVariables.py --era 2017 --region EB1 &
plotIDVariables.py --era 2017 --region EB2 &
plotIDVariables.py --era 2017 --region EE &

plotIDVariables.py --era 2018 --region EB1 &
plotIDVariables.py --era 2018 --region EB2 &
plotIDVariables.py --era 2018 --region EE &