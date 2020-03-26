import torch
from fastai.tabular import LabelLists, TabularList, is_pathlike, defaults
from fastai.basic_data import DatasetType
from fastai.basic_train import load_callback

import pickle

import numpy as np
import pandas as pd

from pathlib import Path
from .config import *
from .utils import to_np

def load_Bravais_models(n_ensembler = N_ESMBLER, which="whole", batch_size = BATCHSIZE):
    models_dir = Path(LEARNER) / Path(BRAVAIS_MODELS_FOLDER[which])
    BE = BravaisEnsembleModel.from_folder(str(models_dir), n=n_ensembler, batch_size=batch_size)
    return BE

def load_Lattice_models(batch_size = BATCHSIZE):
    models_dir = Path(LEARNER) / Path(LATTICE_PARAM_MODELS_FOLDER)
    LPB = LatticeParamModelBundle.from_folder(str(models_dir), batch_size=batch_size)
    return LPB

def load_SpaceGroup_models(batch_size = BATCHSIZE):
    models_dir = Path(LEARNER) / Path(SPACE_GROUP_MODELS_FOLDER)
    SGB = SpaceGroupModelBundle.from_folder(str(models_dir), batch_size=batch_size)
    return SGB

def top_n(preds, n):
    """ return a (preds.shape[0], n) size array """
    if n == 1:
        idxs = np.argmax(preds, axis=1)[:, None]
    else:
        idxs = np.argsort(preds, axis=1)[:, :-n-1:-1]
        
    return idxs

def _pad_sg_out(prob, sg, n): 
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

def complete_lattice_param(out, lps):
    if "cubic" in lps:
        out[["alpha", "beta", "gamma"]] = 90
        out["b"] = out["a"]
        out["c"] = out["a"]
        
    elif "tetragonal" in lps:
        out[["alpha", "beta", "gamma"]] = 90
        out["b"] = out["a"]

    elif "orthorhombic" in lps:
        out[["alpha", "beta", "gamma"]] = 90

    elif "hexagonal" in lps:
        out[["alpha", "beta"]] = 90
        out["gamma"] = 120
        out["b"] = out["a"]

    elif "monoclinic" in lps:
        out[["alpha", "gamma"]] = 90

    elif "triclinic" in lps:
        pass

    elif "rhombohedral" in lps:
        out["beta"] = out['alpha']
        out["gamma"] = out['alpha']
        out["b"] = out['a']
        out["c"] = out['a']
        
    out["v"] = _vol_f[lps](
        out["a"], out["b"], out["c"],
        np.deg2rad(out["alpha"]),
        np.deg2rad(out["beta"]),
        np.deg2rad(out["gamma"]),
    )
    
    return out

class Model:
    def __init__(self, path, file, batch_size):

        # adopted from fastai.load_learner
        source = Path(path)/file if is_pathlike(file) else file
        state = torch.load(source, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(source)
        self.model = state.pop('model')
        self.src = LabelLists.load_state(path, state.pop('data'))
        self.cb_state = state.pop('cb_state')
        self.clas_func = state.pop('cls')
        self.callback_fns = state['callback_fns']
        self.state = state
        self.batch_size = batch_size
                
    def load(self, ext_magpie, **db_kwargs):
        self.src.add_test(TabularList.from_df(ext_magpie), tfm_y=None,)
        data = self.src.databunch(bs=self.batch_size, **db_kwargs)
        res = self.clas_func(data, self.model, **self.state)
        res.callback_fns = self.callback_fns #to avoid duplicates
        res.callbacks = [load_callback(c,s, res) for c,s in self.cb_state.items()]
        self.learn = res
        return self

    def get_preds(self):
        return self.learn.get_preds(ds_type=DatasetType.Test)[0]
    
    def p2o(self, preds, **args):
        return to_np(preds)
        
    def predicts(self, ext_magpie, **args):
        self.load(ext_magpie)
        return self.p2o(self.get_preds(), **args)

class BravaisModel(Model):
    def __init__(self, path, file, batch_size=BATCHSIZE):
        super().__init__(path, file, batch_size)
    
    @property
    def classes(self,):
        return self.learn.data.classes
    
    def p2o(self, preds, n=1, **args):
        preds = super().p2o(preds, **args)
        idxs = top_n(preds, n)
        auxidxs = ( np.tile( np.arange(len(idxs) ), reps=(idxs.shape[1], 1) ) ).T
        return preds[auxidxs, idxs], self.classes[idxs]


    def predicts(self, ext_magpie, topn_bravais = TOPN_BRAVAIS):
        return super().predicts(ext_magpie, n=topn_bravais)

class SpaceGroupModel(Model):
    def __init__(self, path, file, batch_size):
        super().__init__(path, file, batch_size)
    
    @property
    def classes(self,):
        return self.learn.data.classes
    
    def p2o(self, preds, n=1, **args):
        preds = super().p2o(preds, **args)
        idxs = top_n(preds, n)
        auxidxs = ( np.tile( np.arange(len(idxs) ), reps=(idxs.shape[1], 1) ) ).T
        return preds[auxidxs, idxs], self.classes[idxs]


    def predicts(self, ext_magpie, topn_spacegroup = TOPN_SPACEGROUP):
        return super().predicts(ext_magpie, n=topn_spacegroup)

class LatticeParamModel(Model):    
    def __init__(self, path, file, norm, batch_size):
        super().__init__(path, file, batch_size)
        self.norm = norm
    
    def label_denorm(self, preds):
        preds = preds * self.norm['std'] + self.norm['mean']
        return np.exp(preds)
    
    def p2o(self, preds, **args):
        preds = super().p2o(preds, **args)
        return self.label_denorm(preds)

class EnsembleModel:
    
    def __init__(self, Ps):
        self.models = Ps
            
    def load(self, ext_magpie, **db_kwargs):
        for p in self.models:
            p.load(ext_magpie, **db_kwargs)
            
    def get_preds(self):
        return torch.stack([p.get_preds() for p in self.models], dim=0)
    
    def p2o(self, esm_preds, **args):
        return to_np(esm_preds)
    
    def predicts(self, ext_magpie, **args):
        self.load(ext_magpie)
        return self.p2o(self.get_preds(), **args)

class BravaisEnsembleModel(EnsembleModel):

    _esm_prefix = BRAVAIS_ENSEMBLER_PREFIX

    @classmethod
    def from_folder(cls, folder, n=5, batch_size=BATCHSIZE):
        Ps = []
        for i in range(n):
            file = f"{cls._esm_prefix}{i}.pkl"
            Ps.append(BravaisModel(folder, file, batch_size))
        return cls(Ps)
        
    @property
    def classes(self):
        return self.models[0].classes
    
    def p2o(self, esm_preds, n=1, **args):
        # from tensor to numpy
        esm_preds = super().p2o(esm_preds, **args)

        #ensembling strategy: vertical voting
        preds = esm_preds.mean(axis=0)
        idxs = top_n(preds, n=n)
        auxidxs = ( np.tile( np.arange(len(idxs) ), reps=(idxs.shape[1], 1) ) ).T
        return preds[auxidxs, idxs], self.classes[idxs]

    def predicts(self, ext_magpie, topn_bravais = TOPN_BRAVAIS):
        return super().predicts(ext_magpie, n=topn_bravais)
                   
class BLSpliterBundle:
    
    _spliter = BRAVAIS_SPLIT_NAME

    def __init__(self, Ps):
        self.Ps = Ps

    def load(self, ext_magpie_brav):
        groups = ext_magpie_brav.groupby(self._spliter)
        self.idxs = {}
        self.widx = ext_magpie_brav.index
        self.data_size = ext_magpie_brav.shape[0]
        for n, g in groups:
            self.idxs[n] = g.index
            self.Ps[n].load(g.reset_index())
        self.active = self.idxs.keys()

    def get_preds(self):
        return {n: P.get_preds() for n, P in self.Ps.items() if n in self.active }
    
    def p2o(self, preds, **args):
        return {n: self.Ps[n].p2o(pred, **args) for n, pred in preds.items() if n in self.active}
    
    def predicts(self, ext_magpie_brav, **args):
        self.load(ext_magpie_brav)
        return self.p2o(self.get_preds(), **args)


class LatticeParamModelBundle(BLSpliterBundle):
    
    _columns = LATTICE_PARAM_NAMES
    _LatticeParamModels = LATTICE_PARAM_MODELS
    _norms = LATTICE_NORM
    
    @classmethod
    def from_folder(cls, folder, batch_size = BATCHSIZE):
        LPPs = {}
        with (Path(folder)/cls._norms).open("rb") as f:
            norms = pickle.load(f)
            
        for name in BRAVAIS_LATTICE:
            path = cls._LatticeParamModels[name]
            LPPs[name] = LatticeParamModel(folder, path, norms[name], batch_size)
        return cls(LPPs)
    
    def p2o(self, preds, **args):
        preds = super().p2o(preds, **args)
        out = pd.DataFrame(
            np.zeros((self.data_size,7)),
            columns = self._columns+['v'],
            index = self.widx,
        )
        
        for n, idx in self.idxs.items():
            out.loc[idx, PRED_COLS[n]] = preds[n]
            out.loc[idx] = complete_lattice_param(out.loc[idx], n)

        return out


class SpaceGroupModelBundle(BLSpliterBundle):
    
    _SpaceGroupModels = SPACE_GROUP_MODELS
    @classmethod
    def from_folder(cls, folder, batch_size = BATCHSIZE):
        SGPs = {}
        for name in BRAVAIS_LATTICE:
            path = cls._SpaceGroupModels[name]
            SGPs[name] = SpaceGroupModel(folder, path, batch_size)
        return cls(SGPs)
    
    def p2o(self, preds, **args):
        preds = super().p2o(preds, **args)
        # output topN probability and classes 
        # n = args["n"] if "n" in args.keys() else 1
        n = args["n"]
        outs = np.zeros((self.data_size, n) if n > 1 else self.data_size)
        outs_probs = np.zeros((self.data_size, n) if n > 1 else self.data_size)
        for name, idx in self.idxs.items():
            prob, sg = preds[name]
            prob, sg = _pad_sg_out(prob, sg, n)
            outs_probs[idx] = prob
            outs[idx] = sg
        return outs_probs, outs

    def predicts(self, ext_magpie, topn_spacegroup = TOPN_SPACEGROUP):
        return super().predicts(ext_magpie, n=topn_spacegroup)
