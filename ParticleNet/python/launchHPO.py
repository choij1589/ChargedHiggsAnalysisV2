#!/usr/bin/env python
import os
import logging
import argparse
import pandas as pd
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA, MOBSTER
from syne_tune.config_space import lograndint, uniform, loguniform, logfinrange, choice
from syne_tune.utils import streamline_config_space

parser = argparse.ArgumentParser()
parser.add_argument("--signal", required=True, type=str, help="Signal mass point")
parser.add_argument("--background", required=True, type=str, help="Background process")
parser.add_argument("--channel", required=True, type=str, help="Channel")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

# Hyperparameter configuration space
config_space = {
    "signal": args.signal,
    "background": args.background,
    "channel": args.channel,
    "model": "ParticleNet",
    "optimizer": choice(["Adam", "RMSprop", "Adadelta"]),
    "scheduler": choice(["ExponentialLR", "CyclicLR", "ReduceLROnPlateau"]),
    "initLR": loguniform(1e-4, 1e-2),
    "nNodes": lograndint(32, 128),
    "weight_decay": loguniform(1e-5, 1e-2),
    "max_epochs": 81,
}

scheduler = MOBSTER(
    config_space=config_space,
    metric="objective",
    mode="min",
    resource_attr="epoch",
    max_resource_attr="max_epochs",
    grace_period=5,
    reduction_factor=3,
    search_options={"debug_log": True}
)

entry_point = "./python/trainModelSimple.py"
tuner = Tuner(
    trial_backend=LocalBackend(entry_point=entry_point),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(max_wallclock_time=60*60*3),
    n_workers=24
)
tuner.run()

experiment = load_experiment(tuner.name)
results = experiment.results
best_config = pd.DataFrame(experiment.best_config(), index=[0])

outpath = f"results/{args.channel}/ParticleNetV2/CSV/hpo_{args.signal}_vs_{args.background}.csv"
print(f"Best configuration: {best_config}")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
results.to_csv(outpath, index=False)
best_config.to_csv(outpath.replace(".csv", "_best_config.csv"), index=False)
