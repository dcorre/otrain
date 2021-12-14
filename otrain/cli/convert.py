#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: David Corre, IAP, corre@iap.fr

"""

import argparse
import warnings

from otrain.convert import convert

warnings.simplefilter(action="ignore", category=FutureWarning)


def main():

    parser = argparse.ArgumentParser(
        description="Convert simulated data before starting training."
    )

    parser.add_argument(
        "--path",
        dest="path_datacube",
        required=True,
        type=str,
        help="Path where to store the datacube.",
    )

    parser.add_argument(
        "--cube",
        dest="cubename",
        required=True,
        type=str,
        help="Name of the datacube.",
    )

    parser.add_argument(
        "--cutouts",
        dest="path_cutouts",
        required=True,
        type=str,
        help="Path to the cutouts used for the training.",
    )

    parser.add_argument(
        "--frac-true",
        dest="frac_true",
        required=False,
        default=0.5,
        type=float,
        help="Fraction of true candidates to be included in the training set. Flat between 0 and 1.",
    )
    
    parser.add_argument(
        "--cutout-size",
        dest="cutout_size",
        required=False,
        default=10000,
        type=int,
        help="Size of the cutout in pixels, NxN pixels",
    )

    args = parser.parse_args()
    convert(
        args.path_datacube,
        cubename=args.cubename,
        path_cutouts=args.path_cutouts,
        frac_true=args.frac_true,
        cutout_size=args.cutout_size
    )


if __name__ == "__main__":
    main()
