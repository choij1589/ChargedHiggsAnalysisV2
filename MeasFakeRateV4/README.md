# MeasFakeRateV4
---
Use Cut & Count method to measure fake rate of muon and electron

## Procedure
1. Prepare the input file using SKFlatAnalyzer - MeasFakeRateV4
```bash
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeEl8
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeEl12
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeEl23
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeMu8
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeMu17
```
The output of this step will be used for nvtx reweighting in the next step.
2. Run full steps to measure fake rate
Prepare the input file using SKFlatAnalyzer with RunSyst flag
```bash
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeEl8 RunSyst
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeEl12 RunSyst
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeEl23 RunSyst
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeMu8 RunSyst
./script/SubmitAnalyzers/MeasFakeRateV4.sh $ERA MeasFakeMu RunSyst
```
3. Now run the scripts to measure fake rates
```bash
./doThis.sh $ERA $MEASURE
```

4. Need to pass the tests, e.g. MC Closure test, before using the fake rate in the analysis
To run the closure test, use
```bash
./script/SubmitAnalyzers/ClosFakeRate.sh
```
It will submit for all eras

5. Check the systematics in data
It is automatically done when calling ```doThis.sh``` script. You can check the systematic plots in your results.

6. Valid fake rate in Z+fake control region
Additional analyzer required to check the results.
