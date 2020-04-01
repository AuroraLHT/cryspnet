# CRYSPNet

The Crystal Structure Predictions via Neural Network (CRYSPNet) project introduces an alternative way to perform fast estimation on Crystal Structure Information (Bravais Lattice, Space Group, and Latice Parameter) with the power of neural networks. 

## Installation

**Note:** **Python 3.6** or later is required. Since Fastai library does not support Windows, the following installation only works on linux-based environment. We recommand using [CONDA environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create a new environment for this installation.
To install the project with pip and git, run the following commands:
```bash
    cd 
    git clone https://github.com/auroralht/cryspnet.git
    cd cryspnet
    pip install -e .
```

Pre-trained models are stored in google drive. Download the file `learner.zip` from from the [drive](https://drive.google.com/file/d/1s9OkrBRTSWTvufSia-ee625zR73bgBDA/view?usp=sharing). After downing the file copy it to `cryspnet/cryspnet` and extract it. Five folders: `BravaisEsmMetal`, `BravaisEsmOxide`, `BravaisEsmWhole`, `LatticeParam`, and `SpaceGroup` should be in the `cryspnet/cryspnet/learner` directory after the extraction is completed.

## Dependancy

[fastai](https://github.com/fastai/fastai), [pytorch](https://github.com/pytorch/pytorch), and [Matminer](https://hackingmaterials.lbl.gov/matminer/installation.html) are three major package used heavily in this project. If these packages cannot be installed, please go to their github/documentation site for more information.

(optional) We recommand using [JupyterLab](https://github.com/jupyterlab/jupyterlab/tree/acf208ed6f6843d03f34666ffc0cb2c37bdf2f3e) to execute our Notebook example. Running with Jupyter Notebook is extremely fine also. To install:

### conda install
```bash
    conda install -c conda-forge jupyterlab
```

### pip install
```bash
    pip install jupyterlab
```

 is used to generate material descriptor. To install:


## Usage

Input requirement: The input would be stored in a csv like file with
a column name called formula. 

Here is an example of running prediction on formula listed in [demo.csv](https://github.com/auroralht/crystinn/demo/demo.csv)
```bash
    cd cryspnet
    python predict.py -i demo/demo.csv -o output/output.csv 
```

You can also use the Bravais lattice model trained on Metal or Oxide compound by:

```bash
    python predict.py -i demo/demo.csv -o output/output.csv --use_metal
    python predict.py -i demo/demo.csv -o output/output.csv --use_oxide 
```

You could also change the `topn_bravais` and `topn_spacegroup` to see more or less top-N prediction from the Bravais lattice and space group models.
```bash
    python predict.py 
        -i demo/demo.csv \
        -o output/output.csv \
        --topn_bravais 2 \
        --topn_spacegroup 3 \
```

## As Library

The package is wrapper of fastai learner, and is easy to use. The following example shows the basic workfolow:

```python
import pandas as pd
from cryspnet.models import *

formula = pd.read_csv("demo/demo.csv")
BE= load_Bravais_models()
LPB= load_Lattice_models()
SGB= load_SpaceGroup_models()

FG = FeatureGenerator()
predictors = FG(formula)

bravais_probs, bravais = BE.predicts(ext_magpie, topn_bravais=1)
predictors['bravais'] = bravais

spacegroup_probs, spacegroup = SGB.predicts(ext_magpie, topn_spacegroup=1)
latticeparameter = LPB.predicts(ext_magpie)
```

More **examples** could be finded in [Notebook](https://github.com/auroralht/hallucination/demo/).

## History

## Copyright

