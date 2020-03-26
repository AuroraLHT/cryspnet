from crystinn.utils import *
from crystinn.models import *
from crystinn.config import *

import argparse

FG = FeatureGenerator()

def load_input(path):
    path = Path(path)
    if path.suffix == ".csv":
        data = pd.read_csv(str(path), index_col=False)

    elif path.suffix == ".xlsx" or path.suffix == ".xls":
        xls = pd.ExcelFile(path)
        sheets = [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
        data = pd.concat(sheets, axis=0)

    else:
        # we hope the best here
        data = pd.read_csv(path, delimiter= r'\s+', index_col=False, header=None)

    assert data.shape[1] == 1, "the input is not formula and has multiple dimension, plz check the input"
    data.columns = ['formula']
    return data

def dump_output(output, path, **args):
    output.to_csv(path, **args)

def group_outputs(bravais, bravais_probs, spacegroups, spacegroups_probs, lattices, formula):
    topn_bravais = bravais.shape[1]
    topn_spacegroup = spacegroups[0].shape[1]

    inner_columns =  ["Bravais", "Bravais prob"] + \
        LATTICE_PARAM_NAMES + ['v'] + \
        [f"Top-{i+1} SpaceGroup" for i in range(topn_spacegroup) ] + \
        [f"Top-{i+1} SpaceGroup prob" for i in range(topn_spacegroup) ]

    idxs = [('formula', "-")]+ [ (f"Top-{i+1} Bravais", c) for i in range(topn_bravais) for c in inner_columns ]

    idxs = pd.MultiIndex.from_tuples(idxs)
    out = pd.DataFrame(columns = idxs)

    out['formula'] = formula['formula']

    for i in range(topn_bravais):
        out[f"Top-{i+1} Bravais", "Bravais"] = bravais[:, i]
        out[f"Top-{i+1} Bravais", "Bravais prob"] = bravais_probs[:, i]
        for j in range(topn_spacegroup):
            out[f"Top-{i+1} Bravais", f"Top-{j+1} SpaceGroup"] = spacegroups[i][:, j].astype(int)
            out[f"Top-{i+1} Bravais", f"Top-{j+1} SpaceGroup prob"] = spacegroups_probs[i][:, j]
        out.loc[:, (f"Top-{i+1} Bravais", lattices[i].columns) ]  =  lattices[i].values

    return out


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-i", "--input", default=None, type=str, required=True,
                        help="The input data path. The program accept .csv, .xlsx file.")
    parser.add_argument("-o", "--output", default=None, type=str, required=True,
                        help="The output directory where predictions for \
                            Bravais Lattice, Space Group, and Lattice  will be written.")
    parser.add_argument("--use_metal", action='store_true',
                        help="Whether to run prediction on the Bravais Lattice model that trained on metal subset.")
    parser.add_argument("--use_oxide", action='store_true',
                        help="Whether to run prediction on the Bravais Lattice model that trained on oxide subset.")
    parser.add_argument("--n_ensembler", default=N_ESMBLER, type=int,
                        help="number of ensembler for Bravais Lattice Prediction.")
    parser.add_argument("--topn_bravais", default=TOPN_BRAVAIS, type=int,
                        help="The top-n Bravais Lattice the user want to pre \
                        serve. The space group and lattice parameter would \
                        be predicted for each top-n Bravais Lattice"
            )
    parser.add_argument("--topn_spacegroup", default=TOPN_SPACEGROUP, type=int,
                        help="The top-n Space Group the user want to pre \
                        serve."
            )
    parser.add_argument("--batch_size", default=BATCHSIZE, type=int,
                        help="Batch size per GPU/CPU for prediction.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    args = parser.parse_args()

    if args.no_cuda:
        defaults.device = torch.device('cpu')
    if args.use_metal and args.use_oxide:
        raise Exception("Could only select --use_metal or --use_oxide")
    elif args.use_metal:
        which = "metal"
    elif args.use_oxide:
        which = "oxide"
    else:
        which = "whole"

    BE = load_Bravais_models(
            n_ensembler = args.n_ensembler,
            which = which,
            batch_size = args.batch_size)
    LPB = load_Lattice_models(batch_size = args.batch_size)
    SGB = load_SpaceGroup_models(batch_size = args.batch_size)

    formula = load_input(args.input)
    ext_magpie = FG.generate(formula)

    bravais_probs, bravais = BE.predicts(ext_magpie, topn_bravais=args.topn_bravais)

    lattices = []
    spacegroups = []
    spacegroups_probs = []

    for i in range(args.topn_bravais):
        ext_magpie["Bravais"] = bravais[:, i]
        lattices.append(LPB.predicts(ext_magpie))
        sg_prob, sg = SGB.predicts(ext_magpie, topn_spacegroup=args.topn_spacegroup)
        spacegroups.append(sg)
        spacegroups_probs.append(sg_prob)

    out = group_outputs(bravais, bravais_probs, spacegroups, spacegroups_probs, lattices, formula)
    dump_output(out, args.output, index=False)

if __name__ == "__main__":
    main()
