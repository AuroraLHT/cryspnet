import torch
# from fastai.tabular import LabelLists, TabularList, is_pathlike, defaults
# from fastai.basic_data import DatasetType
# from fastai.basic_train import load_callback
from fastai.tabular.all import load_learner, accuracy

import pickle

import numpy as np
import pandas as pd

from pathlib import Path
from .config import *
from .utils import to_np, topkacc

from typing import Tuple, Dict, Union, List

def load_Bravais_models(n_ensembler:int = N_ESMBLER, which:str="whole", batch_size:int = BATCHSIZE, cpu:bool=False)->torch.nn.Module:
    models_dir = Path(LEARNER) / Path(BRAVAIS_MODELS_FOLDER[which])
    BE = BravaisEnsembleModel.from_folder(str(models_dir), n=n_ensembler, batch_size=batch_size, cpu=cpu)
    return BE

def load_Lattice_models(batch_size:int = BATCHSIZE, cpu:bool=False)->torch.nn.Module:
    models_dir = Path(LEARNER) / Path(LATTICE_PARAM_MODELS_FOLDER)
    LPB = LatticeParamModelBundle.from_folder(str(models_dir), batch_size=batch_size, cpu=cpu)
    return LPB

def load_SpaceGroup_models(batch_size:int = BATCHSIZE, cpu:bool=False)->torch.nn.Module:
    models_dir = Path(LEARNER) / Path(SPACE_GROUP_MODELS_FOLDER)
    SGB = SpaceGroupModelBundle.from_folder(str(models_dir), batch_size=batch_size, cpu=cpu)
    return SGB

def top_n(preds:np.ndarray, n:int)->np.ndarray:
    """ return a (preds.shape[0], n) size array """
    if n == 1:
        idxs = np.argmax(preds, axis=1)[:, None]
    else:
        idxs = np.argsort(preds, axis=1)[:, :-n-1:-1]
        
    return idxs

def _pad_sg_out(prob:np.ndarray, sg:np.ndarray, n:int)->Tuple[np.ndarray, np.ndarray]: 
    sgfill=-1
    filltype=int

    if (n==1 and len(prob.shape)==1 and len(sg.shape==1)):
        prob = prob[:, None]
        sg = sg[:, None]

    if prob.shape[1]!=n:
        residual = n - prob.shape[1]
        pad_prob = np.zeros((prob.shape[0], residual))
        pad_sg = np.empty((prob.shape[0], residual), dtype=filltype)
        pad_sg.fill(sgfill)
        prob = np.concatenate((prob,pad_prob), axis=1)
        sg = np.concatenate((sg,pad_sg), axis=1)

    return prob, sg
 

_vol_f = {
    "cubic (P)" : lambda a, b, c, alpha, beta, gamma : a**3,
    "cubic (F)" : lambda a, b, c, alpha, beta, gamma : a**3,
    "cubic (I)" : lambda a, b, c, alpha, beta, gamma : a**3,
    
    "monoclinic (P)": lambda a, b, c, alpha, beta, gamma : a*b*c*np.sin(beta),
    "monoclinic (C)": lambda a, b, c, alpha, beta, gamma : a*b*c*np.sin(beta),

    "hexagonal (P)": lambda a, b, c, alpha, beta, gamma : np.sqrt(3)/2 * a**2 * c,
    
    "rhombohedral (P)": lambda a, b, c, alpha, beta, gamma : a**3*np.sqrt(1 - 3*np.cos(alpha)**2 + 2*np.cos(alpha)**3),
    
    "tetragonal (P)" : lambda a, b, c, alpha, beta, gamma : a**2*c,
    "tetragonal (I)" : lambda a, b, c, alpha, beta, gamma : a**2*c,

    "orthorhombic (P)" : lambda a, b, c, alpha, beta, gamma : a*b*c,
    "orthorhombic (C)" : lambda a, b, c, alpha, beta, gamma : a*b*c,
    "orthorhombic (F)" : lambda a, b, c, alpha, beta, gamma : a*b*c,
    "orthorhombic (I)" : lambda a, b, c, alpha, beta, gamma : a*b*c,
    

    "triclinic (P)" : lambda a, b, c, alpha, beta, gamma : a*b*c*np.sqrt(1-np.cos(alpha)**2- \
              np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
}

def complete_lattice_param(out:pd.DataFrame, bravais:str):
    if "cubic" in bravais:
        out[["alpha", "beta", "gamma"]] = 90
        out["b"] = out["a"]
        out["c"] = out["a"]
        
    elif "tetragonal" in bravais:
        out[["alpha", "beta", "gamma"]] = 90
        out["b"] = out["a"]

    elif "orthorhombic" in bravais:
        out[["alpha", "beta", "gamma"]] = 90

    elif "hexagonal" in bravais:
        out[["alpha", "beta"]] = 90
        out["gamma"] = 120
        out["b"] = out["a"]

    elif "monoclinic" in bravais:
        out[["alpha", "gamma"]] = 90

    elif "triclinic" in bravais:
        pass

    elif "rhombohedral" in bravais:
        out["beta"] = out['alpha']
        out["gamma"] = out['alpha']
        out["b"] = out['a']
        out["c"] = out['a']
        
    out["v"] = _vol_f[bravais](
        out["a"], out["b"], out["c"],
        np.deg2rad(out["alpha"]),
        np.deg2rad(out["beta"]),
        np.deg2rad(out["gamma"]),
    )
    
    return out

class Model:
    """
        base class for loading a single pytorch model from pre-trained weights
    """

    def __init__(self, file_name:Union[Path, str], batch_size:int, cpu:bool=True):
        # adopted from fastai.load_learner
        self.learn = load_learner(file_name, cpu=cpu)
        self.batch_size = batch_size

    def load(self, ext_magpie:pd.DataFrame, **db_kwargs):
        # adopted from fastai.load_learner
        dl = self.learn.dls.test_dl(ext_magpie, bs=self.batch_size)
        self.dl = dl
        return self

    def get_preds(self):
        return self.learn.get_preds(dl=self.dl)[0]
    
    def p2o(self, preds:torch.Tensor, **args):
        return to_np(preds)
        
    def predicts(self, ext_magpie:pd.DataFrame, **args):
        self.load(ext_magpie)
        return self.p2o(self.get_preds(), **args)

class BravaisModel(Model):
    """A single model for predicting Bravais Lattice"""

    def __init__(self, file_name:Union[Path, str], batch_size:int=BATCHSIZE, cpu:bool=False):
        super().__init__(file_name, batch_size, cpu=cpu)
    
    @property
    def classes(self,):
        # return self.learn.data.classes
        return self.learn.classes # stored previously by training script
    
    def p2o(self, preds:torch.Tensor, n:int=1, **args)->Tuple[np.ndarray, np.ndarray]:
        preds = super().p2o(preds, **args)
        idxs = top_n(preds, n)
        auxidxs = ( np.tile( np.arange(len(idxs) ), reps=(idxs.shape[1], 1) ) ).T
        return preds[auxidxs, idxs], self.classes[idxs]

    def predicts(self, ext_magpie:pd.DataFrame, topn_bravais:int=TOPN_BRAVAIS):
        return super().predicts(ext_magpie, n=topn_bravais)

class SpaceGroupModel(Model):
    """A single model for predicting Space Group"""

    def __init__(self, file_name:Union[Path, str], batch_size:int, cpu:bool=False):
        super().__init__(file_name, batch_size, cpu=cpu)
    
    @property
    def classes(self,):
        return self.learn.classes
    
    def p2o(self, preds:torch.Tensor, n:int=1, **args)->Tuple[np.ndarray, np.ndarray]:
        preds = super().p2o(preds, **args)
        idxs = top_n(preds, n)
        auxidxs = ( np.tile( np.arange(len(idxs) ), reps=(idxs.shape[1], 1) ) ).T
        return preds[auxidxs, idxs], self.classes[idxs]


    def predicts(self, ext_magpie:pd.DataFrame, topn_spacegroup:int = TOPN_SPACEGROUP):
        return super().predicts(ext_magpie, n=topn_spacegroup)

class LatticeParamModel(Model):
    """A single model for predicting Lattice Parameters"""

    def __init__(self, filename:Union[Path, str], norm:Dict[str, np.ndarray], batch_size:int, cpu:bool=False):
        super().__init__(filename, batch_size, cpu=cpu)
        self.norm = norm
    
    def label_denorm(self, preds:torch.Tensor):
        preds = preds * self.norm['std'] + self.norm['mean']
        return np.exp(preds)
    
    def p2o(self, preds:torch.Tensor, **args):
        preds = super().p2o(preds, **args)
        return self.label_denorm(preds)

class EnsembleModel:
    """A base class for applying ensembling on prediction from many models"""
    
    def __init__(self, Ms:List[Model]):
        self.models = Ms
            
    def load(self, ext_magpie:pd.DataFrame, **db_kwargs):
        for m in self.models:
            m.load(ext_magpie, **db_kwargs)
            
    def get_preds(self):
        return torch.stack([m.get_preds() for m in self.models], dim=0)
    
    def p2o(self, esm_preds:torch.Tensor, **args):
        return to_np(esm_preds)
    
    def predicts(self, ext_magpie:pd.DataFrame, **args):
        self.load(ext_magpie)
        return self.p2o(self.get_preds(), **args)

class BravaisEnsembleModel(EnsembleModel):
    """A class for merging prediction from many Bravais Lattice models"""

    _esm_prefix = BRAVAIS_ENSEMBLER_PREFIX

    @classmethod
    def from_folder(cls, folder:Union[Path, str], n:int=5, batch_size:int=BATCHSIZE, cpu:bool=False):
        Ms = []
        folder = Path(folder)
        if not folder.exists(): raise FileNotFoundError(str(folder))

        for i in range(n):
            filename =  Path(folder) / f"{cls._esm_prefix}{i}.pkl"
            if not folder.exists(): raise FileNotFoundError(str(filename))
            Ms.append(BravaisModel( filename, batch_size, cpu=cpu))
        return cls(Ms)
        
    @property
    def classes(self):
        return self.models[0].classes
    
    def p2o(self, esm_preds:torch.Tensor, n:int=1, **args):
        # from tensor to numpy
        esm_preds = super().p2o(esm_preds, **args)

        #ensembling strategy: vertical voting
        preds = esm_preds.mean(axis=0)
        idxs = top_n(preds, n=n)
        auxidxs = ( np.tile( np.arange(len(idxs) ), reps=(idxs.shape[1], 1) ) ).T
        return preds[auxidxs, idxs], self.classes[idxs]

    def predicts(self, ext_magpie:pd.DataFrame, topn_bravais:int = TOPN_BRAVAIS):
        return super().predicts(ext_magpie, n=topn_bravais)
                   
class BLSpliterBundle:
    """ A base class that split the input data by Bravias Lattice and send each group to the corresponding model"""

    _spliter = BRAVAIS_SPLIT_NAME

    def __init__(self, Ms:Dict[str, Model]):
        self.Ms = Ms

    def load(self, ext_magpie_brav:pd.DataFrame):
        groups = ext_magpie_brav.groupby(self._spliter)
        self.idxs = {}
        self.widx = ext_magpie_brav.index
        self.data_size = ext_magpie_brav.shape[0]
        for n, g in groups:
            self.idxs[n] = g.index
            self.Ms[n].load(g.reset_index())
        self.active = self.idxs.keys()

    def get_preds(self):
        return {n: P.get_preds() for n, P in self.Ms.items() if n in self.active }
    
    def p2o(self, preds:Dict[str, torch.Tensor], **args):
        return {n: self.Ms[n].p2o(pred, **args) for n, pred in preds.items() if n in self.active}
    
    def predicts(self, ext_magpie_brav:pd.DataFrame, **args):
        self.load(ext_magpie_brav)
        return self.p2o(self.get_preds(), **args)


class LatticeParamModelBundle(BLSpliterBundle):
    """ A class that store Lattice Parameters models for each Bravais Lattice and provide a unified API to predict Lattice Parameters"""

    _columns = LATTICE_PARAM_NAMES
    _LatticeParamModels = LATTICE_PARAM_MODELS
    _norms = LATTICE_NORM
    
    @classmethod
    def from_folder(cls, folder:Union[Path, str], batch_size:int = BATCHSIZE, cpu:bool=False):
        LPMs = {}

        folder = Path(folder)
        if not folder.exists() : raise FileExistsError(folder)

        with (folder/cls._norms).open("rb") as f:
            norms = pickle.load(f)

        for name in BRAVAIS_LATTICE:
            path = cls._LatticeParamModels[name]
            filename = folder / path
            if not filename.exists() : raise FileExistsError(filename)
            LPMs[name] = LatticeParamModel(filename, norms[name], batch_size, cpu=cpu)
        return cls(LPMs)
    
    def p2o(self, preds:Dict[str, torch.Tensor], **args):
        preds = super().p2o(preds, **args)
        out = pd.DataFrame(
            np.zeros((self.data_size, 7)),
            columns = self._columns+['v'],
            index = self.widx,
        )
        
        for n, idx in self.idxs.items():
            out.loc[idx, PRED_COLS[n]] = preds[n]
            out.loc[idx] = complete_lattice_param(out.loc[idx], n)

        return out

class SpaceGroupModelBundle(BLSpliterBundle):
    """ A class that store Lattice Parameters models for each Bravais Lattice and provide a unified API to predict Lattice Parameters"""

    _SpaceGroupModels = SPACE_GROUP_MODELS
    @classmethod
    def from_folder(cls, folder:Union[Path, str], batch_size:int = BATCHSIZE, cpu:bool=False):
        SGMs = {}
        folder = Path(folder)
        if not folder.exists(): raise FileExistsError(folder)

        for name in BRAVAIS_LATTICE:
            path = cls._SpaceGroupModels[name]
            filename = folder / path
            if not filename.exists(): raise FileExistsError(filename)
            SGMs[name] = SpaceGroupModel(filename, batch_size, cpu=cpu)
        return cls(SGMs)
    
    def p2o(self, preds:Dict[str, torch.Tensor], **args):
        preds = super().p2o(preds, **args)
        # output topN probability and classes 
        # n = args["n"] if "n" in args.keys() else 1
        n = args["n"]
        outs = np.zeros((self.data_size, n))
        outs_probs = np.zeros((self.data_size, n))
        for name, idx in self.idxs.items():
            prob, sg = preds[name]
            prob, sg = _pad_sg_out(prob, sg, n)
            outs_probs[idx] = prob
            outs[idx] = sg
        return outs_probs, outs

    def predicts(self, ext_magpie:pd.DataFrame, topn_spacegroup:int = TOPN_SPACEGROUP):
        return super().predicts(ext_magpie, n=topn_spacegroup)
