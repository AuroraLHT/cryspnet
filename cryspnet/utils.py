from functools import partial
from collections.abc import Iterable
from collections import defaultdict
import re
from pathlib import Path


# from IPython.display import display
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from sklearn.metrics import confusion_matrix

import torch
from fastai.basic_data import DatasetType
from fastai.basic_train import load_learner

# import plotly
# import plotly.plotly as py
import plotly.graph_objs as go

# Author logan
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition

import Equation # math expression parser


# pytorch stuff
def to_np(x):
    "Convert a tensor to a numpy array."
    return x.data.cpu().numpy()

ELEMENTS = set(['Pu', 'Re', 'Y', 'Bk', 'S', 'Hf', 'Br', 'Eu', 'Al', 'Li',
    'Md', 'Sm', 'Be', 'B', 'No', 'Te', 'Kr', 'Co', 'P', 'Cu', 'N', 'Ac',
    'Nd','Yb', 'Gd', 'Tb', 'Es', 'Fr', 'Th', 'Si', 'Zr', 'Na', 'Pd', 'U',
    'Ni', 'Rn', 'H', 'Cl', 'Au', 'Lu', 'Pr', 'Pa', 'In', 'Er', 'Mn', 'I',
    'Ne', 'Os', 'Mg', 'O', 'Ga', 'F', 'Sr', 'Ru', 'Bi', 'Dy', 'Ra', 'Ho',
    'Xe', 'Tm', 'As', 'Am', 'Ir', 'Hg', 'Sc', 'Cd', 'Cr', 'Se', 'Ta',
    'Fm', 'Rb', 'Sn', 'Tc', 'Rh', 'Lr', 'Np', 'Pm', 'Pb', 'Ca', 'Cs',
    'Nb', 'Ag', 'V', 'He', 'Zn', 'Mo', 'Ti', 'Sb', 'Fe', 'Ge', 'Po', 'La',
    'Tl', 'Ba', 'Ce', 'C', 'Cm', 'Cf', 'Pt', 'W', 'K', 'Ar', 'At'])

NON_METAL_ELEMENTS = set(["H", "He", "B", "C", "N", "O", "F", "Ne",
                         "Si", "P", "S", "Cl", "Ar", "Ge", "As",
                         "Se", "Br", "Kr", "Sb", "Te", "I",
                         "Xe", "Po", "At", "Rn", "Ts", "Og"])

METAL_ELEMENTS = ELEMENTS - NON_METAL_ELEMENTS

def is_oxide(x):
    return len(re.findall("O[0-9]", x))>0

def is_metal(x):
    return set(re.findall(r"([A-Za-z]+)[0-9\.]*", x)).issubset(METAL_ELEMENTS)

def has_metal(x):
    return len(set(re.findall(r"([A-Za-z]+)[0-9\.]*", x)) & METAL_ELEMENTS)>0

# plotting ternary
def ternary_trace(x, y, z, type="scatter", color=None, name=None, size=23, prop=0.9, symbol="hexagon", note=None, colorscale=None, colorbar=None, cmin=None, cmax=None, addline=False):
    """
        x, y, z : scatter points coordinate
        ptype : type of ternary plot (scatterternary, )
        color : color values
        name : target class name
        size : size of the marker
        prop : the probability of the prediction
        symbol : marker type
        note : (str) any knowledge needed to add to the records
        colorscale : colormap name
        colorbar : setting for colorbar
        cmin, cmax : limit for color values
    """
    trace = dict(
        type = "scatterternary",
        a = x,
        b = y,
        c = z,
        name = name,
    )
    
    if type == "scatter":
        trace.update(dict(
            mode = "markers",
            marker = dict(
                opacity = prop,
                symbol = symbol,
                size = size,
                color= color,
                cmin = cmin,
                cmax = cmax,
                colorscale=colorscale)
            ))
        if colorbar is not None:
            trace["marker"]['colorbar'] = colorbar
        if isinstance(prop, Iterable):
            trace.update( text = list(map(lambda x:"conf: {:.3f}".format(x), prop)))
        if note is not None:
            trace.update( text = note )
        if addline:
            trace["marker"].update(line=dict(width=2,color='DarkSlateGrey'),)
    else:
        trace.update(dict(
            mode='lines',
            line=dict(color='#444'),
            fill='toself',
            fillcolor=color
            )
        )
    return trace

def ternary_layout(title=None, xtitle=None, ytitle=None, ztitle=None, xr=(0, 1), yr=(0, 1), zr=(0, 1), legend=None):
    layout = dict(
        title = title,
        ternary = dict(
            sum = 1,
            aaxis = {'title':xtitle, 'min':0, 'linewidth':2, 'ticks':'outside'},
            baxis = {'title':ytitle, 'min':0, 'linewidth':2, 'ticks':'outside'},
            caxis = {'title':ztitle, 'min':0, 'linewidth':2, 'ticks':'outside'},
        ),
        showlegend = True,
    )
    if legend is not None: layout['legend'] = legend
    return layout

# tools for material search
def tri_grid(n, xr=(0,1), yr=(0,1), zr=(0,1)):
    """
    generate a ternary "grip" for ternary component search
    use the term "triangle" might be more proper

    xr : x range
    yr : y range
    zr : z range
    n : the number of element for first axis 
    - the total number of the grid is (n*n-1)/2
    - you can imagine it as the base length of a triangle
    """
    for i, c1 in enumerate(np.linspace(xr[0], xr[1], n)):
        for c2 in np.linspace(yr[0], min(1-c1, yr[1]), n-i):
            if 1-c1-c2 >= zr[0] and 1-c1-c2 <= zr[1]:
                yield (c1, c2, 1-c1-c2)



VAR_SYMBOLS = {"x", "y", "z"}

def get_vars(comp_formulas):
    """find the variable in the formulas"""
    character_set = set("".join(comp_formulas))
    return sorted(list(character_set.intersection(VAR_SYMBOLS)))

class Element:
    def __init__(self, name, comp):
        self.name = name
        self.comp = comp

# matminer wrapper to generate features 
class Compound:
    def __init__(self, *elements):
        self.elements = elements
        self.vars = get_vars([ele.comp for ele in self.elements])
        self.eles = [ele.name for ele in self.elements]
    
    @classmethod
    def from_str(cls, formula):
        inps = re.findall("([a-zA-Z]+)\s?([x-z/0-9*\(\)\-\.]*)", formula)
        elements = [Element(*inp) for inp in inps]
        return cls(*elements)

    def feagen(self, limits=None, n=50):
        if len(self.elements) == 3 and len(self.vars) == 0:
            if limits is None: limits = [(0,1), (0,1), (0,1)]
            fg = SingleCompFeatureGenerator(n=n)
            return fg.ternary(compound=self, limits=limits)
        else:
            assert limits is not None, "plz set the search limits to each variable"
            fg = SingleCompFeatureGenerator(n=n)
            return fg.formula_search(compound=self, limits=limits)

    def __getitem__(self, key):
        return self.elements[key]
    
    def __str__(self):
        return " ".join(["{} {}".format(ele.name, ele.comp) for ele in self.elements])


class CompoundBundle:
    def __init__(self, *compounds):
        self.compounds = compounds
        self.eles = sorted(set(ele.name for comp in compounds for ele in comp.elements))
        
    def __getitem__(self, key):
        return self.compounds[key]


class FeatureGenerator:
    def __init__(self):
        self.feature_calculators = MultipleFeaturizer([
            cf.ElementProperty.from_preset(preset_name="magpie"),
            cf.Stoichiometry(),
            cf.ValenceOrbital(props=['frac']),
            cf.IonProperty(fast=True),
            cf.BandCenter(),
            cf.ElementFraction(),
        ])

        self.str2composition = StrToComposition()
        
    def generate(self, fake_df, ignore_errors=False):
        """
            generate feature from a dataframe with a "formula" column that contains 
            chemical formulas of the compositions.
        """
        fake_df = self.str2composition.featurize_dataframe(fake_df, "formula", ignore_errors=ignore_errors)
        fake_df = fake_df.dropna()
        fake_df = self.feature_calculators.featurize_dataframe(fake_df, col_id='composition', ignore_errors=ignore_errors);
        fake_df["NComp"] = fake_df["composition"].apply(len)
        return fake_df

class MultiCompFeatureGenerator(FeatureGenerator):
    def __init__(self, n=50):
        super().__init__()
        self.n = n

    def ternary_dope(self, compounds, limits):
        eles = compounds.eles
        concentrations = np.stack(list( tri_grid(self.n, *limits)), axis=0)
        x, y, z = concentrations[:,0,None], concentrations[:,1,None], concentrations[:,2,None]
        bases = []
        for compound in compounds:
            base = np.zeros(len(eles))
            for ele in compound:
                base[eles.index(ele.name)] = ele.comp
            bases.append(base)

        concentrations = x*bases[0] + y*bases[1] + z*bases[2]
        composition_format = "{:.2f}".join(eles) + "{:.2f}"
        formula = [composition_format.format(*row) for row in concentrations]
        fake_df = pd.DataFrame({"formula":formula, "x": x.ravel(), "y": y.ravel(), "z": z.ravel()})
        return self.generate(fake_df)

class SingleCompFeatureGenerator(FeatureGenerator):
    def __init__(self, n=50):
        super().__init__()
        self.n = n
    
    def ternary(self, compound, limits):
        """ generate a search list for the ternary compound """ 
        
        vars = compound.vars
        eqs = [Equation.Expression(ele.comp, compound.vars) for ele in compound.elements ]

        eles = [ ele.name for ele in compound.elements ]
        composition_format = "{:.2f}".join(eles) + "{:.2f}"
        
        concentrations = np.stack(list( tri_grid(self.n, *limits)), axis=0)
        formula = [composition_format.format(*row) for row in concentrations]
        fake_df = pd.DataFrame({"formula":formula, "x": concentrations[:,0], "y": concentrations[:,1], "z": concentrations[:,2]})
        return self.generate(fake_df)

    def formula_search(self, compound, limits):
        """ generate a search list for the compound with variable in it"""
        variable_search_lists = [np.linspace(start=limit[0], stop=limit[1], num=self.n ) for limit in limits]
        search_sheet = {var: variable_search_list for var, variable_search_list in zip(compound.vars, variable_search_lists)}

        # generate the composition given the variable using the equation from Compound
        compositions = [eq(*variable_search_lists) for eq in self.eqs]
        compositions = [np.repeat(comp, self.n) if not isinstance(comp, np.ndarray) else comp for comp in compositions]
        
        composition_format = "{:.2f}".join(eles) + "{:.2f}"
        search_sheet['formula'] = [composition_format.format(*row) for row in np.stack(compositions, axis=1)]
        return self.generate(pd.DataFrame(search_sheet))
    
def alias_lookup(name):
    
    alias_book = defaultdict(lambda : None,
    cubic_F = "fcc",
    cubic_I = "bcc",
    tetragonal_I = "bct",
    hexagonal = "hcp"
    )
    
    alias = alias_book[name]
    if alias is not None: return alias
    else: return name    