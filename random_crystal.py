import argparse
import pickle
from pathlib import Path
import time
import re
from multiprocessing import Pool, TimeoutError
from functools import partial
from tqdm import tqdm
import logging

import pandas as pd
import numpy as np

from pyxtal.structure import Xstruct
from pyxtal.crystal import random_crystal, Lattice

import pymatgen

from typing import Dict, List, Union

from cryspnet.utils import LATTICE_PARAM_ERROR, LATTICE_PARAM_MODELS_FOLDER, LEARNER

DEFAULT_ERROR = str( Path(LEARNER) / Path(LATTICE_PARAM_MODELS_FOLDER) / Path(LATTICE_PARAM_ERROR) )

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def is_valid_crystal(rc:random_crystal):
    return (rc is not None) and rc.valid

def sample_lattice(one:pd.Series, bra:str, trails:int, err_dict:Dict=None):
    """
        Use error from the model's validation set to add variation in the predicted lattice
        If the err_dict is supplied then lattice parameter is simplity copied by "trails" times.

        Arguments:
            one : one row from the input dataframe
            bra : Bravais Lattice
            trails : number of lattice we want to produce for crystal generation
            err_dict : error from the lattice parameter model
    """

    lt = bra.split(" ")[0]
    if err_dict is None:
        a,b,c,alpha,beta,gamma = one[['a','b', 'c', 'alpha', 'beta', 'gamma']]
        a,b,c,alpha,beta,gamma = float(a), float(b), float(c), float(alpha),float(beta),float(gamma)

        for i in range(trails):
            yield Lattice.from_para(a, b, c, alpha, beta, gamma, ltype=lt,)

    else:
        if "cubic" in bra:
            random_error = np.random.normal(err_dict[bra]['mean'], err_dict[bra]['std'], trails)

            min_random_err = err_dict[bra]['mean'] - 2*err_dict[bra]['std']
            max_random_err = err_dict[bra]['mean'] + 2*err_dict[bra]['std']

        else:
            random_error = np.random.multivariate_normal(err_dict[bra]['mean'], err_dict[bra]['std'], trails)

            min_random_err = err_dict[bra]['mean'] - 2*np.diagonal(err_dict[bra]['std'])
            max_random_err = err_dict[bra]['mean'] + 2*np.diagonal(err_dict[bra]['std'])

        random_error = np.clip(random_error, min_random_err, max_random_err)

        for i in range(trails):
            if "cubic" in bra:
                a = one['a'] / random_error[i]
                yield Lattice.from_para(a, a, a, 90, 90, 90, ltype=lt,)

            elif "tetragonal" in bra:
                a, c = one[['a', 'c']] / random_error[i]
                yield Lattice.from_para(a, a, c, 90, 90, 90, ltype=lt,)
                
            elif "orthorhombic" in bra:
                a, b, c = one[['a', 'b', 'c']] / random_error[i]
                yield Lattice.from_para(a, b, c, 90, 90, 90, ltype=lt,)

            elif "hexagonal" in bra:
                a, c = one[['a', 'c']] / random_error[i]
                yield Lattice.from_para(a, a, c, 90, 90, 120, ltype=lt,)

            elif "monoclinic" in bra:
                a, b, c, beta = one[['a', 'b', 'c', 'beta']] / random_error[i]
                yield Lattice.from_para(a, b, c, 90, beta, 90, ltype=lt,)

            elif "triclinic" in bra:
                a, b, c, alpha, beta, gamma = one[['a', 'b', 'c', 'alpha', 'beta', 'gamma']] / random_error[i]
                yield Lattice.from_para(a, b, c, alpha, beta, gamma, ltype=lt,)

            elif "rhombohedral" in bra:
                a, alpha= one[['a', 'alpha']] / random_error[i]
                yield Lattice.from_para(a, a, a, alpha, alpha, alpha, ltype=lt,)

def is_stoi(formula:str):
    """ check if the formula contain fractional stoichiometric """
    return "." not in formula

def decomp(formula:str):
    """ parse formula into elements and stoichiometric """
    groups = re.findall("([A-Za-z]{1,2})([0-9]+)", formula)
    
    elements, stois = list( zip(*groups) )
    stois = list(map(int, stois))
    return elements, stois

def try_random_crystal(formula:str, sg:int, elements:List[str], stois:Union[List[int], np.ndarray], lattice:Lattice=None, vf:float=1.0, max_multi:int=5, max_atoms:int=50, start:int=-1):
    """
        Try to generate a crystal from given space group, elements, stoichiometric, and lattice (optional) information.
        Due to mechanism of Pyxtal, the input stoichiometric might need to be multiplied by a integar. The max_multi set the 
        maximum value the integar could be. max_atom set the maximum atoms a lattice could have. 

        Arguments:
            formula : chemical formula
            sg : space group number 
            elements : a list of elements symbol
            stois : a list or array of stoichoimetric values
            lattice : a Lattice object that contains lattice parameters and crystal system (optional)
            vf : volume factor (see random_crystal from Pyxtal)
            max_multi : maximum value of multiplicity would be tried on
            max_atoms : maximum value of atoms in a lattice
            start : multiplicity that explicitly given to work on, if this one is given then the method would not try other multiplicity
    """
    
    def _try(formula:str, sg:int, elements:List[str], stois:Union[List[int], np.ndarray], lattice:Lattice, vf:float):
        try:
            # logging.debug( f"{formula} {sg} {elements} {stois} {lattice} {vf}")
            crystal = random_crystal(sg, elements, stois, vf, lattice=lattice)
            logging.debug(f"_try Is crystal valid {is_valid_crystal(crystal)}" )
        except Exception as e:
            logging.error(f"During random crystal generation: \n {formula} {sg} {elements} {stois} \n Error Message: {e}")
            crystal = None
        finally:
            return crystal

    elements = list(elements)
    stois = np.array(stois)

    if start==-1:
        # try multiply the input stoichiometric from 1 to max_multiplicity to see if the wyckoff postion is compatible
        for multi in range(1, max_multi+1):
            if max_atoms >= np.sum(stois) * multi:
                crystal= _try(formula, int(sg), list(elements), list(stois*multi), lattice, vf)
                logging.debug(f"_try Is crystal valid {is_valid_crystal(crystal)}" )
                if is_valid_crystal(crystal): return crystal, multi
        return None, -1
    else:
        crystal=_try(formula, sg, list(elements), list(stois*start), lattice, vf)
        if is_valid_crystal(crystal) : return crystal, start
        return None, -1

def get_max_topn_bravais(df:pd.DataFrame):
    """get the number of bravais lattice stored in the dataframe"""

    l1 = list(df.columns.get_level_values(0))
    l1 = [ int(c.split(" ")[0].split("-")[1]) for c in l1 if "Bravais" in c]
    return max(l1)

def get_max_topn_spacegroup(df:pd.DataFrame):
    """get the number of space group stored in the dataframe"""

    l2 = list(df.columns.get_level_values(1))
    l2 = [ int(c.split(" ")[0].split("-")[1]) for c in l2 if "SpaceGroup" in c]
    return max(l2)

def save_random_crystal(rc:random_crystal, path:str):
    """ save the generated crystal to cif format"""

    rc.to_file(fmt='cif', filename=path)
    logging.debug(f"Save rc to --> {path}")

def process(one:pd.Series, output:Path, n_trails:int, topn_bravais:int, topn_spacegroup:int, max_atoms:int, err_dict:dict):
    """
        Generate crystals from a given row of the input DataFrame (produced by CRYSPNet).
        The number of generated crystals is control by "n_trails". 
        If error is given then lattice would be varied for each trails.
        The generated crystal would be save in folder "output" with the format "formula_spacegroup_trails.cif" 

        Arguments:
            one : a row from CRYSPNet prediction
            output : the folder the generated crystals to be saved at
            n_trails : maximum number of crystals that the process would generated
            topn_bravais : select prediction from 1 to top-n bravais as input
            topn_spacegroup : select prediction from 1 to top-n space group as input
            max_atoms : maximum amount of atoms in a lattice
            err_dict : a dictionary of error term of the lattice parameters model (from CRYSPNet)
    """

    start_t = time.time() 

    formulas = []
    paths = []
    formula = one['formula']['-']
    elements, stois = decomp(formula)

    for topn_b in range(1, topn_bravais+1):
        bra = one[f'Top-{topn_b} Bravais']['Bravais']
        for topn_sg in range(1, topn_spacegroup+1):
            mul = -1

            sg = int(one[f'Top-{topn_b} Bravais'][f'Top-{topn_sg} SpaceGroup'])
            # sg = int(one[f'Top-{topn_b} Bravais'][f'Top-{topn_b} SpaceGroup']) # don't delete this, it is a reminder of a historic bug caused by copy and paste, please check !
            # importance of logging each step

            for trail, lattice in enumerate(sample_lattice(one[f'Top-{topn_b} Bravais'], bra, n_trails, err_dict=err_dict)):
                rc, mul = try_random_crystal(formula, sg, elements, stois, lattice=lattice, start=mul, max_atoms=max_atoms)

                logging.debug(f"Process is_valid_crystal {is_valid_crystal(rc)}")
                if is_valid_crystal(rc):
                    path = output/f"{formula}_{sg}_{trail}.cif"
                    save_random_crystal(rc, path)

                    formulas.append(formula)
                    paths.append(path)
                else:
                    logging.info(f"{formula} maximum trail exceed at trail {trail} break")
                    break

    end_t = time.time()
    logging.info(f"Finished {formula} in {end_t - start_t:.1f}s")
    return formulas, paths

def process_space_group_only(one:pd.Series, output:Path, n_trails:int, topn_bravais:int, topn_spacegroup:int, max_atoms:int):
    """
        Generate crystals from a given row of the input DataFrame (produced by CRYSPNet).
        Lattice parameter information is not used in this method.
        The number of generated crystals is control by "n_trails". 
        If error is given then lattice would be varied for each trails.
        The generated crystal would be save in folder "output" with the format "formula_spacegroup_trails.cif" 

        Arguments:
            one : a row from CRYSPNet prediction
            output : the folder the generated crystals to be saved at
            n_trails : maximum number of crystals that the process would generated
            topn_bravais : select prediction from 1 to top-n bravais as input
            topn_spacegroup : select prediction from 1 to top-n space group as input
            max_atoms : maximum amount of atoms in a lattice
    """
    
    
    formulas = []
    paths = []
    formula = one['formula']['-']
    elements, stois = decomp(formula)

    for topn_b in range(1, topn_bravais+1):
        bra = one[f'Top-{topn_b} Bravais']['Bravais'].split(" ")[0]
        for topn_sg in range(1, topn_spacegroup+1):
            mul = -1
            sg = int(one[f'Top-{topn_b} Bravais'][f'Top-{topn_sg} SpaceGroup'])

            for trail in range(n_trails):
                rc, mul = try_random_crystal(sg, elements, stois, lattice=None, start=mul, max_atoms=max_atoms)
                if is_valid_crystal(rc): break
                path = output/f"{formula}_{sg}_{trail}.cif"
                save_random_crystal(rc, path)

                formulas.append(formula)
                paths.append(path)

    logging.info(f"finished {formula}")
    return formulas, paths

def process_formula_only(one:pd.Series, output:Path, n_trails:int, max_atoms:int):
    """
        Generate crystals from a given row of the input DataFrame (any DataFrame has the same format as CRYSPNet).
        The number of generated crystals is control by "n_trails". 
        If error is given then lattice would be varied for each trails.
        The generated crystal would be save in folder "output" with the format "formula_spacegroup_trails.cif" 

        Arguments:
            one : a row from CRYSPNet prediction
            output : the folder the generated crystals to be saved at
            n_trails : maximum number of crystals that the process would generated
            max_atoms : maximum amount of atoms in a lattice
    """
    formulas = []
    paths = []

    formula = one['formula']['-']
    elements, stois = decomp(formula)

    for sg in range(1, 230+1):
        mul = -1

        for trail in range(n_trails):
            rc, mul = try_random_crystal(sg, elements, stois, lattice=None, start=mul)
            if is_valid_crystal(rc): 
                path = output/f"{formula}_{sg}_{trail}.cif"
                save_random_crystal(rc, path)

                formulas.append(formula)
                paths.append(path)
            else:
                logging.info(f"{formula} maximum trail exceed at trail {trail} break")

    logging.info(f"finished {formula}")
    return formulas, paths

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-i", "--input", default=None, type=str, required=True,
                        help="The input data path. The program accept .csv, .xlsx file."
    )
    parser.add_argument("-e", "--error", default=DEFAULT_ERROR, type=str, required=False,
                        help="The error associated with the prediction"
    )
    parser.add_argument("-o", "--output", default=None, type=str, required=True,
                        help="The output directory where predictions for \
                            Bravais Lattice, Space Group, and Lattice  will be written."
    )
    parser.add_argument("--topn_bravais", default=2, type=int,
                        help="The top-n Bravais Lattice the user want to pre \
                        serve. The space group and lattice parameter would \
                        be predicted for each top-n Bravais Lattice"
    )
    parser.add_argument("--topn_spacegroup", default=1, type=int,
                        help="The top-n Space Group the user want to pre \
                        serve."
    )
    parser.add_argument("--n_workers", default=4, type=int,
                        help="Number of workers used to generate random crystal"
    )
    parser.add_argument("--n_trails", default=100, type=int,
                        help="Number of trails for a given composition, space group, and lattice parameter"
    )
    parser.add_argument("--timeout", default=100, type=int,
                        help="You ultimate patient level in the unit of second! some entries would just run forever so we have to do discard it"
    )
    parser.add_argument("--formula_only", action='store_true',
                        help="Use Only formula as information to generate structure"
    )
    parser.add_argument("--space_group_only", action='store_true',
                        help="Use Only the space group information but not lattice parameter to generate structure"
    )
    parser.add_argument("--max_atoms", default=50, type=int,
                        help="the maximum number of atoms per unit cell, setted to avoid generating superlarge unit cell that slow down the calculation"
    )

    args = parser.parse_args()

    if args.error is not None and Path(args.error).exists():
        with open(args.error, "rb") as f:
            err_dict = pickle.load(f)
        logging.info(f"use error from {args.error}")
    else:
        logging.info("do not use error")
        err_dict = None

    csv = pd.read_csv(args.input,  header=[0,1])
    stoi_entries = csv.loc[ csv['formula']['-'].map(is_stoi) ]

    output = Path(args.output)
    output.mkdir(exist_ok=True)
    
    topn_bravais = min(args.topn_bravais, get_max_topn_bravais(stoi_entries))
    topn_spacegroup = min(args.topn_spacegroup, get_max_topn_spacegroup(stoi_entries))

    if args.n_workers <= 0:
        raise Exception("argument number of worker is less than 1")
    elif args.n_workers == 1:
        logging.info(f"use single process")
        for i, row in tqdm(stoi_entries.iterrows(), total=len(stoi_entries)):
            if args.formula_only:
                f, p = process_formula_only(row, output=output, n_trails=args.n_trails, max_atoms=args.max_atoms)
            elif args.space_group_only:
                f, p = process_space_group_only(row, output=output, n_trails=args.n_trails, topn_bravais=topn_bravais, topn_spacegroup=topn_spacegroup, max_atoms=args.max_atoms)
            else:
                f, p = process(row, output=output, n_trails=args.n_trails, topn_bravais=topn_bravais, topn_spacegroup=topn_spacegroup, max_atoms=args.max_atoms, err_dict=err_dict)
    else:
        logging.info(f"use multiprocess with {args.n_workers} worders")
        if args.formula_only:
            f = partial(process_formula_only, output=output, n_trails=args.n_trails, max_atoms=args.max_atoms)
        elif args.space_group_only:
            f = partial(process_space_group_only, output=output, n_trails=args.n_trails, topn_bravais=topn_bravais, topn_spacegroup=topn_spacegroup, max_atoms=args.max_atoms)
        else:
            f = partial(process, output=output, n_trails=args.n_trails, topn_bravais=topn_bravais, topn_spacegroup=topn_spacegroup, max_atoms=args.max_atoms, err_dict=err_dict)
        
        with Pool(processes=args.n_workers) as pool:
            async_list = []
            formula_list = []
            for i, row in stoi_entries.iterrows():
                formula = row['formula']['-']
                formula_list.append(formula)
                res = pool.apply_async(f, (row,)) # this (row, ) could be further changed by chunksize
                async_list.append(res)

            for res, formula in zip(async_list, formula_list):
                try:
                    fs, ps  = res.get(timeout=args.timeout)
                except TimeoutError:
                    logging.info(f"{formula} timeout in {args.timeout}s!")
                except Exception as e:
                    logging.error(f"Other Error encounter {e}")
                finally:
                    pass

    all_cifs = list(output.glob("*.cif"))
    formulas = list(map(lambda x: x.name.split('_')[0], all_cifs))

    pd.DataFrame({
        "formula" : formulas,
        "path" : list(map(lambda x: x.name, all_cifs)),
    }).to_csv(str(output/"index.csv"), index=False)

    logging.info("Index file is saved to --> {}".format(str(output/"index.csv")))

if __name__ == "__main__":
    main()
