import os
import uuid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--signal", required=True, type=str, help="signal mass point")
parser.add_argument("--background", required=True, type=str, help="background")
parser.add_argument("--channel", required=True, type=str, help="Skim1E2Mu / Skim3Mu / Combined")
args = parser.parse_args()

process = uuid.uuid4().hex.upper()[:6]
condorBase = f"condor/{args.channel}/{args.signal}_vs_{args.background}_{process}"
os.makedirs(condorBase, exist_ok=True)

def makeCondorSub():
    f = open(f"{condorBase}/condor.sub", "w")
    f.write(f"executable = {condorBase}/run.sh\n")
    f.write(f"jobbatchname = ParticeNet_{args.signal}_vs_{args.background}_{args.channel}\n")
    f.write('+singularityimage = "/data9/Users/choij/Singularity/images/cuda11.8"\n')
    f.write("requirements = HasSingularity\n")
    f.write("request_disk = 100 GB\n")
    f.write("request_memory = 100 GB\n")
    f.write("request_cpus = 20\n")
    f.write("request_gpus = 1\n")
    f.write(f"log = {condorBase}/job.log\n")
    f.write(f"output = {condorBase}/job.out\n")
    f.write(f"error = {condorBase}/job.err\n")
    f.write("queue 1")
    f.close()

def makeRunSh():
    f = open(f"{condorBase}/run.sh", "w")
    f.write("#/bin/bash\n")
    f.write("source /opt/conda/bin/activate\n")
    f.write("conda activate pyg\n")
    f.write('export WORKDIR="/data6/Users/choij/ChargedHiggsAnalysisV2"\n')
    f.write('export PATH="${PATH}:${WORKDIR}/ParticleNet/python"\n')
    #f.write('export PYTHONPATH="${PYTHONPATH}:${WORKDIR}/python"\n')
    f.write("cd $WORKDIR/ParticleNet\n")
    f.write(f"launchHPO.py --signal {args.signal} --background {args.background} --channel {args.channel}\n")
    f.close()

if __name__ == "__main__":
    print(f"Running condor job in {condorBase}")
    makeCondorSub()
    makeRunSh()
    os.chmod(f"{condorBase}/run.sh", 0o755)
    os.system(f"condor_submit {condorBase}/condor.sub")
