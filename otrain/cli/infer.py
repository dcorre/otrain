#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: David Corre, IAP, corre@iap.fr

"""

import argparse
import warnings

from otrain.infer import infer

warnings.simplefilter(action="ignore", category=FutureWarning)


def main():

    parser = argparse.ArgumentParser(
        description="Apply a previously trained model to a dataset."
    )

    parser.add_argument(
        "--cutouts",
        dest="path_cutouts",
        required=True,
        type=str,
        help="Path to cutouts.",
    )

    parser.add_argument(
        "--prob-ratio",
        dest="probratio",
        required=False,
        default=0.001,
        type=float,
        help="Proba ratio, not used at the moment.",
    )

    parser.add_argument(
        "--model",
        dest="path_model",
        required=True,
        type=str,
        help="Path to the trained model, including its name and extension.",
    )

    args = parser.parse_args()
    infer(args.path_cutouts, args.path_model, args.probratio)


if __name__ == "__main__":
    main()
