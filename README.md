# ChargedHiggsAnalysisV2

## Abstract
In the context of the Two-Higgs-Doublet Model, this analysis investigates the decay of the charged Higgs boson into a CP-odd Higgs boson and a W boson, using proton-proton collision data at 13\TeV recorded by the CMS detector during LHC Run-2. The focus centers on decay sequences starting with a top quark decaying into a charged Higgs and a bottom quark, followed by the charged Higgs boson decaying into a W boson and a CP-odd Higgs boson, which then decays into two muons. Event selection targets oppositely-signed muon pairs in electron-muon-muon or muon-muon-muon configurations, accompanied by jets and b-jets from top quark decays. ParticleNet, a graph-based deep neural network, is utilized to enhance event identification, particularly for muon pairs with mass signatures similar to Z boson decays.
[AN-23-203](https://gitlab.cern.ch/tdr/notes/AN-23-203)

## How to
In most cases, submodules require SKFlatOutput for data and ROOT for the analysis framework, which can run natively using C++ and python. In the previous version of ChargedHiggsAnalysis, running the analysis differed module by module. In V2, we try to incorporate the usage of each module, using cmake to seamlessly integrate the core C++ classes and functions to ROOT framework, and use python for scripting and automatizing each modules. The only exceptions will be two submodules - pytorch framework and Higgs Combine Tools. Those frameworks will be controlled using different git repositories, since the configurations will be different with other modules. More details can be found in each README.md files in the submodules.

```bash
# Initial starting point
source setup.sh
```

## Install some necessary packages
```bash
# install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py

# set conda env
source /opt/conda/bin/activate 
# if the directory does not exist, install miniconda following the link below
# https://docs.anaconda.com/miniconda/#quick-command-line-install
conda update -n base -c defaults conda
conda config --set channel_priority strict
conda create -n pyg -c conda-forge python=3.11 root=6.30.04 -y
conda activate pyg
#add below commands in ~/.bashrc
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# install torch & torch_geometric
# used cu121 instead of cu118 to make the version of CUDA and PyTorch compatible
#pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
#pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# install other packages
pip install cmake
pip install numpy pandas matplotlib scipy scikit-learn seaborn captum networkx
#pip install torchlars
pip install torchlars --no-cache-dir
git clone https://github.com/awslabs/syne-tune.git
cd syne-tune
pip install -e '.[extra]'
````
