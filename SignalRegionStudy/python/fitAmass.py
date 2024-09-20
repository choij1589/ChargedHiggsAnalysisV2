import os
import logging
import argparse
import pandas as pd
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

WORKDIR = os.getenv("WORKDIR")
