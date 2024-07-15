#!/bin/bash
export PATH="${PWD}/python:${PATH}"

for era in 2016preVFP 2016postVFP 2017 2018; do
    # 1E2Mu
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey pair/mass --rebin 5
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey electrons/1/pt
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey muons/1/pt
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey muons/2/pt
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey electrons/1/scEta --rebin 5
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey muons/1/eta --rebin 4
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey muons/2/eta --rebin 4
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey nonprompt/pt --rebin 5
    plotClosure.py --era ${era} --channel Skim1E2Mu --histkey nonprompt/eta --rebin 5

    plotSystematics2.py --era ${era} --channel Skim1E2Mu --histkey pair/mass
    
    # 3Mu
    plotClosure.py --era ${era} --channel Skim3Mu --histkey stack/mass --rebin 5
    plotClosure.py --era ${era} --channel Skim3Mu --histkey muons/1/pt
    plotClosure.py --era ${era} --channel Skim3Mu --histkey muons/2/pt
    plotClosure.py --era ${era} --channel Skim3Mu --histkey muons/3/pt
    plotClosure.py --era ${era} --channel Skim3Mu --histkey muons/1/eta --rebin 4
    plotClosure.py --era ${era} --channel Skim3Mu --histkey muons/2/eta --rebin 4
    plotClosure.py --era ${era} --channel Skim3Mu --histkey muons/3/eta --rebin 4
    plotClosure.py --era ${era} --channel Skim3Mu --histkey nonprompt/pt --rebin 5
    plotClosure.py --era ${era} --channel Skim3Mu --histkey nonprompt/eta --rebin 4

    plotSystematics2.py --era ${era} --channel Skim3Mu --histkey stack/mass
done
