#!/usr/bin/env python
import os, shutil
import logging
import argparse
import subprocess
from scipy.stats import loguniform
from time import sleep
from GATools import GeneticModule

parser = argparse.ArgumentParser()
parser.add_argument("--signal", required=True, type=str, help="signal mass point")
parser.add_argument("--background", required=True, type=str, help="background process")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--device", required=True, type=str, help="which device to use, cpu or cuda:0...")
parser.add_argument("--nPop", type=int, default=16, help="population size")
parser.add_argument("--maxIter", type=int, default=5, help="max iteration")
parser.add_argument("--fold", required=True, type=str, help="fold number for the training, 0...nFolds-1")
parser.add_argument("--pilot", action="store_true", default=False, help="pilot mode")
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
args = parser.parse_args()

WORKDIR = os.getenv("WORKDIR")
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

# Set up base working directory
base_dir = f"{WORKDIR}/dataset/{args.channel}/{args.signal}_vs_{args.background}_fold-{args.fold}"
if args.pilot:
    base_dir = f"{WORKDIR}/dataset/{args.channel}__pilot__/{args.signal}_vs_{args.background}_fold-{args.fold}"
if os.path.exists(base_dir):
    # raise question to user, are you sure to delete the model?
    print(f"{base_dir} exists, delete it? [y/n]")
    answer = input()
    if answer == "y":
        logging.info(f"Deleting base directory {base_dir}")
        shutil.rmtree(base_dir)
    else:
        logging.info(f"Abort")
        exit(0)

# Config Space setting
nNodes = [64, 96, 128]
optimizers = ["RMSprop", "Adam", "Adadelta"]
schedulers = ["ExponentialLR", "CyclicLR", "ReduceLROnPlateau"]
initLRs = [round(value, 4) for value in loguniform.rvs(1e-4, 1e-2, size=100)]
weight_decays = [round(value, 5) for value in loguniform.rvs(1e-5, 1e-2, size=1000)]
#initLRs = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
#weight_decays = [0., 0.0005, 0.001, 0.005, 0.01, 0.05]
thresholds = [0.5, 0.5, 0.5, 0.5, 0.5]

## Communication function for the GeneticModule and single training process
def evalFitness(population, iteration):
    procs = []
    for idx in range(args.nPop):
        if population[idx]["fitness"]:
            # fitness already estimated
            continue
        nNodes, optimizer, initLR, weight_decay, scheduler = population[idx]["chromosome"]
        command = f"python/trainSglConfigForGA.py --signal {args.signal} --background {args.background}"
        command += f" --channel {args.channel}"
        command += f" --iter {iteration} --idx {idx}"
        command += f" --nNodes {nNodes} --optimizer {optimizer} --initLR {initLR} --scheduler {scheduler}"
        command += f" --weight_decay {weight_decay} --device {args.device} --fold {args.fold}"
        if args.pilot:
            command += f" --pilot"
        logging.info(f"Start training {idx}th model with command: {command}")
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(proc)
        sleep(0.5)

    for proc in procs:
        stdout, stderr = proc.communicate()
        #print("Output:", stdout.decode())
        #print("Errors:", stderr.decode())
    
        assert proc.returncode == 0, f"Process failed with return code {proc.returncode}"

# generate pool
gaModule = GeneticModule()
gaModule.setConfigSpace(nNodes)
gaModule.setConfigSpace(optimizers)
gaModule.setConfigSpace(initLRs)
gaModule.setConfigSpace(weight_decays)
gaModule.setConfigSpace(schedulers)
gaModule.generatePopulation()

gaModule.randomGeneration(n_population=args.nPop)
evalFitness(gaModule.population, iteration=0)
gaModule.updatePopulation(args.signal, args.background, args.channel, iteration=0)
logging.info(f"Generation 0")
logging.info(f"Best chromosome: {gaModule.bestChromosome()}")
logging.info(f"Mean fitness: {gaModule.meanFitness()}")
csv_path = f"{base_dir}/GA-iter0/CSV/model_info.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
gaModule.savePopulation(csv_path)
for i in range(1, args.maxIter):
    gaModule.evolution(thresholds=thresholds, ratio=1) # new pool is 
    evalFitness(gaModule.population, iteration=i)
    gaModule.updatePopulation(args.signal, args.background, args.channel, iteration=i)
    logging.info(f"Generation {i}")
    logging.info(f"Best chromosome: {gaModule.bestChromosome()}")
    logging.info(f"Mean fitness: {gaModule.meanFitness()}")
    csv_path = f"{base_dir}/GA-iter{i}/CSV/model_info.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    gaModule.savePopulation(csv_path)
