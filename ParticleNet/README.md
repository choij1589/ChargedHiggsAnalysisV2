# ParticleNet
---
## Introduction
In this module, several scripts are used:
- Prepare dataset: ```saveDatasets.sh```
- launch Genetic Algorithm for hyperparameter optimization: ```launchGAOptim.sh```
- re-train models with the best hyperparameters, for each fold of dataset: ```evalModels.sh```
- after training, best resuls are parsed to /results folder: ```parseBestResults.sh```
- finally, results are paresed to SKFlat: ```toSKFlat.sh```

## Current Status
- Training with loosened signal regions (no b-jet requirement, using loose lepton IDs)
- No transformation in training dataset
- Using 5-fold cross-validation
