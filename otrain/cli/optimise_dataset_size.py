#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: David Corre, IAP, corre@iap.fr

"""

import argparse
import warnings

from otrain.optimise_dataset_size import optimise_dataset_size

warnings.simplefilter(action="ignore", category=FutureWarning)


def main():

    parser = argparse.ArgumentParser(
        description="Train the CNN on different dataset sizes"
    )

    parser.add_argument(
        "--cube",
        dest="path_cubename",
        required=True,
        type=str,
        help="Path to the datacube, including its name and extension.",
    )

    parser.add_argument(
        "--model-path",
        dest="path_model",
        required=True,
        type=str,
        help="Path where to store the trained model.",
    )

    parser.add_argument(
        "--model-name",
        dest="modelname",
        required=True,
        type=str,
        help="Name of the trained model.",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        required=False,
        type=int,
        default=10,
        help="Number of epochs. (Default: 10)",
    )

    parser.add_argument(
        "--n_sections",
        dest="frac",
        required=False,
        type=int,
        default=20,
        help="The number of sections we want to divide the dataset into."
        "(Default: 20)",
    )

    parser.add_argument(
        "--dropout",
        dest="dropout",
        required=False,
        type=float,
        default=0.3,
        help="Fraction used for each dropout. " "(Default: 0.3)",
    )

    parser.add_argument(
        "--outdir",
        dest="outdir",
        required=True,
        type=str,
        help="Path where to store the results.",
    )

    args = parser.parse_args()
    optimise_dataset_size(
        args.path_cubename,
        args.path_model,
        args.modelname,
        args.epochs,
        n_sections=args.frac,
        dropout=args.dropout,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
