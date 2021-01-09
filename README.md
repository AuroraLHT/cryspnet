# CRYSPNet

The Crystal Structure Prediction Network ([CRYSPNet](hhttps://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.123802)) project introduces an alternative way to perform fast prediction on Crystal Structure Information (Bravais Lattice, Space Group, and Lattice Parameter) with the power of neural networks. 

## Installation

**Note:** **Python 3.6** or later is required. Since Fastai library does not support Windows, the following installation only works on a linux-based environment. We recommend using [CONDA environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to create a new environment for this installation.
To install the project with pip and git, run the following commands:
```bash
    git clone https://github.com/auroralht/cryspnet.git
    cd cryspnet
    pip install -e .
```

Pre-trained models are stored in google drive. Download the file `learner.zip` from the [drive](https://drive.google.com/file/d/1rpbV2-mnNj3M16-4BKvhuo5pkeoIY96q/view?usp=sharing). After downloading the file, pls copy it to `cryspnet/cryspnet` and extract it. Five folders: `BravaisEsmMetal`, `BravaisEsmOxide`, `BravaisEsmWhole`, `LatticeParam`, and `SpaceGroup` should be in the `cryspnet/cryspnet/learner` directory after the extraction is completed.

## 🔥Update🔥 
The library has moved from fastai v1 to fastai v2; thus Bravais Lattice, Lattice Parameters and Space Group models are retrained. Please **download** the latest models from [here](https://drive.google.com/file/d/1rpbV2-mnNj3M16-4BKvhuo5pkeoIY96q/view?usp=sharing). The link to the [old](https://drive.google.com/file/d/1s9OkrBRTSWTvufSia-ee625zR73bgBDA/view?usp=sharing) version.

To update the library itself:
```bash
    cd cryspnet
    git pull
    pip install -r requirements.txt
```

## ⚠️About fastai⚠️
*If you are not interested in training your model, then you can skip this part*

*This section would be removed after they fixed this issue in the next release version*

The current fastai v2 tabular modulus has data leakage issues when trying to export the learner. The fix is only made in the development repository. To install:
```bash
    pip uninstall fastai
    git clone https://github.com/fastai/fastai
    pip install -e "fastai[dev]"
```

## Dependancy

[fastai](https://github.com/fastai/fastai), [pytorch](https://github.com/pytorch/pytorch), and [Matminer](https://hackingmaterials.lbl.gov/matminer/installation.html) are three major package used heavily in this project. Please go to their GitHub/documentation site for more information if these packages cannot be installed.

(optional) We recommend using [JupyterLab](https://github.com/jupyterlab/jupyterlab/tree/acf208ed6f6843d03f34666ffc0cb2c37bdf2f3e) to execute our Notebook example. Running with Jupyter Notebook is extremely fine also. To install:

### conda install
```bash
    conda install -c conda-forge jupyterlab
```

### pip install
```bash
    pip install jupyterlab
```

(⚠️ISSUE⚠️) When running through the notebook, a tqdm issue might raise, saying IProcess is not found. It could be solved by installing the [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html) 

### conda install
```bash
    conda install -c conda-forge ipywidgets
```

### pip install
```bash
    pip install ipywidgets
```


## Usage

Input requirement: The input would be stored in a csv file with
a column name called formula. 

Here is an example of predicting the Bravais, space group, and lattice parameter of formula listed in [demo.csv](https://github.com/AuroraLHT/cryspnet/tree/master/demo)
```bash
    cd cryspnet
    python predict.py -i demo/demo.csv -o output/output.csv 
```

You can also use the Bravais lattice model trained on Metal or Oxide compounds by:

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

The output is stored in .csv format with the first two rows as headers. The layout of output is shown by this example:
| formula | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais | Top-1 Bravais |
| ------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------  | ------- |------ | ------------- | ------------- |
| -      | Bravais | Bravais prob | a | b | c | alpha | beta | gamma | v | Top-1 SpaceGroup | Top-2 SpaceGroup | Top-1 SpaceGroup prob | Top-2 SpaceGroup prob |
| Co15Ga2Pr2 | rhombohedral (P)        | 0.847  | 6.50 | 6.50 | 6.50 | 86.4 | 86.4 | 86.4 | 274 | 166 | 160 | 0.98 | 0.01 |

The first row is the major index that groups various predictions of one Bravais Lattice type together. The first column, "**formula**" like its name said, shows the chemical formula. "**Top-n Bravais**" shows this part of the prediction is from the n-th most likely Bravais Lattice. In the second row, "**Bravais**" and "**Bravais prob**" shows the Bravais Lattice and its predicted probability. "**a**", "**b**", "**c**", "**alpha**", "**beta**", "**gamma**" show the predicted lattice parameters and "**v**" shows the unit cell volume. "**Top-k Spacegroup**" and "**Top-k Spacegroup prob**" show the k-th most likely spacegroup number and its predicted probability.

To open this .csv in python, consider using these lines:
```python
    import pandas as pd
    pd.read_csv("output/output.csv",  header=[0,1])
```

## As Library

The package is a wrapper of fastai learner, and is easy to use. The following example shows the basic workflow:

```python
    import pandas as pd
    from cryspnet.models import *

    formula = pd.read_csv("demo/demo.csv")
    BE = load_Bravais_models()
    LPB = load_Lattice_models()
    SGB = load_SpaceGroup_models()

    featurizer = FeatureGenerator()
    predictors = featurizer.generate(formula)

    bravais_probs, bravais = BE.predicts(predictors, topn_bravais=1)
    predictors['bravais'] = bravais

    spacegroup_probs, spacegroup = SGB.predicts(predictors, topn_spacegroup=1)
    latticeparameter = LPB.predicts(predictors)
```

More **examples** could be finded in [Notebook](https://github.com/AuroraLHT/cryspnet/tree/master/Notebook).

## 🌟Train Your Own CRYSPNET🌟
We provide three notebooks: [TrainBravais](https://github.com/AuroraLHT/cryspnet/tree/master/Notebook/TrainBravais.ipynb), [TrainSpaceGroup](https://github.com/AuroraLHT/cryspnet/tree/master/Notebook/TrainSpaceGroup.ipynb), and [TrainLattice](https://github.com/AuroraLHT/cryspnet/tree/master/Notebook/TrainLattice.ipynb) to showcase the training process of each component. 


## Randan Crystal Generation with PyXtal

[PyXtal](https://github.com/qzhu2017/PyXtal) is an open-source library that could generate structures from chemical formulas and space group inputs. Combining with this library, here we demonstrate a workflow to create candidate crystal structures for only chemical formula input. To install PyXtal:

### pip install
```bash
    pip install pyxtal==0.0.8
```

Here is an example of using PyXtal in conjunct with the prediction from CRYSPNet:
```bash
    python predict.py -i demo/demo.csv -o output/output.csv 

    python random_crystal.py \
        -i output/output.csv \
        -o output/random_crystal \
        --topn_bravais 1 \
        --topn_spacegroup 1 \
        --n_workers 4 \
        --n_trails 100 
```
