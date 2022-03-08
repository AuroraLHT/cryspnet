from cryspnet.utils import FeatureGenerator, load_input, dump_output, group_outputs, topkacc
from cryspnet.models import load_Bravais_models, load_Lattice_models, load_SpaceGroup_models
from cryspnet.config import *

import argparse

featurizer = FeatureGenerator()

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

    use_cpu = args.no_cuda
    
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
            batch_size = args.batch_size,
            cpu=use_cpu
    )
    LPB = load_Lattice_models(batch_size = args.batch_size, cpu=use_cpu)
    SGB = load_SpaceGroup_models(batch_size = args.batch_size, cpu=use_cpu)

    formula = load_input(args.input)
    ext_magpie = featurizer.generate(formula)

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
