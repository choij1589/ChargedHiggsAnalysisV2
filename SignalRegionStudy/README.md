# SignalRegionStudy

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
scramv1 b clean; scramv1 b -j 12 # always make a clean build
```

## Preprocess
First, we have to same input shapes for Combine. In SKFlat, I have already skimmed SignalRegion events,
only need to run scripts/preprocess.sh to make Combine input shapes.
```bash
./scripts/preprocess.sh $ERA $CHANNEL
```
Here, CHANNEL is Skim1E2Mu or Skim3Mu

## Prepare datacards and input shapes
Currently, only supporting for shape analysis without peak fitting.
```bash
./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT
```
Here, $CHANNEL is SR1E2Mu or SR3Mu
