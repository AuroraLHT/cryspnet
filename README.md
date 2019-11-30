# Hallucination

The Hallucination project introduces an alternative way to perform fast estimation on Crystal Structure Information (Bravais Lattice, Space Group, and Latice Parameter) with the power of neural networks. 

## Installation

**Note:** **Python 3.6** or later is required. We recommand using [CONDA environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
To install the project with pip, run the follow command:
```bash
pip install -e git+https://github.com/auroralht/hallucination
```

Pre-trained models are stored in google drive. Download the file `learner.zip` from from the [drive](https://drive.google.com/file/d/1s9OkrBRTSWTvufSia-ee625zR73bgBDA/view?usp=sharing). After downing the file copy it to `hallucination/hallucination` and extract it. Five folders: `BravaisEsmMetal`, `BravaisEsmOxide`, `BravaisEsmWhole`, `LatticeParam`, and `SpaceGroup` should be in the `hallucination/hallucination/learner` directory after the extraction is completed.

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

Here is an example of running prediction on formula listed in [demo.csv](https://github.com/auroralht/hallucination/demo/demo.csv)
```bash
    cd hallucination
    python predict.py -i demo/demo.csv -o output/output.csv 
```

A more detailed argument is shown below:
```bash
    python predict.py  \
        -i demo/demo.csv \ # input directory
        -o output/output.csv \ # output directory
        --use_metal \ # use models trained only on metal entries
        --use_oxide \ # use models trained only on oxide entries
        --n_ensembler 5 \ # number of models used to predict Bravais
        --topn_bravais 2 \ # get top-n result (Bravais Lattice)
        --topn_spacegroup 3 \ # get top-n result (Space Group)
        --batch_size 256 \ # batchsize, reduce if leak of memory 
        --no_cuda \ # forcely run on cpu
```

## As Library

The package is wrapper of fastai learner, and is easy to use. The following example shows the basic workfolow:

```python
import pandas as pd
from hallucination.models import *

formula = pd.read_csv("demo/demo.csv")
BE= load_Bravais_models()
LPB= load_Lattice_models()
SGB= load_SpaceGroup_models()

FG = FeatureGenerator()
ext_magpie = FG(formula)

bra_probs, bravais = BE.predicts(ext_magpie, topn_bravais=1)
ext_magpie['bravais'] = bravais

sg_probs, sg = SGB.predicts(ext_magpie, topn_spacegroup=1)
lp = LPB.predicts(ext_magpie)
```

More **examples** could be finded in [Notebook](https://github.com/auroralht/hallucination/demo/).

## History

## Copyright

