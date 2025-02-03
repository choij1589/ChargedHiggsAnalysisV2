# SignalRegionStudyV1

## Installation steps for Combine
Combine should be installed locally in ChargedHiggsAnalysisV2/CommonTools directory. 
We will use Combine v10, which is recommended to use with CMSSW\_14\_1\_0\_pre4.
```bash
# In ChargedHiggsAnalysisV2/CommonTools
export SCRAM_ARCH=el8_amd64_gcc12
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv

# Install combine
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit

cd $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v10.0.2

cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
cd CombineHarvester
git checkout v3.0.0-pre1

cd $CMSSW_BASE/src
scramv1 b clean; scramv1 b -j 12 # always make a clean build
```

## Now all the necessary scripts are based on ERA / CHANNEL / MASSPOINT / METHOD level.
## For the masspoints that are not to be optimized, can directly make templates and datacards.
```bash
preprocess.py --era $ERA --channel $CHANNEL --signal $MASSPOINT # here, $CHANNEL is Skim1E2Mu or Skim3Mu
makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
checkTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD \
    >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt
```

## For the masspoints that are to be optimized, you should update scores
```bash
preprocess.py --era $ERA --channel $CHANNEL --signal $MASSPOINT # here, $CHANNEL is Skim1E2Mu or Skim3Mu
makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method ParticleNet --update
checkTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method ParticleNet
printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method ParticleNet \
    >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/ParticleNet/datacard.txt
```
Then the updated scores will be stored in /samples directory. To rerun, just erase the files in /samples and rerun preprocess.
```

## Prepare datacards and input shapes
Currently, only supporting for template based shape analysis.
```bash
./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT
```
Here, $CHANNEL is SR1E2Mu or SR3Mu
