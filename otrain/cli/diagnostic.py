#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: David Corre, IAP, corre@iap.fr

"""

import argparse
import warnings

from otrain.diagnostics import print_diagnostics, get_diagnostics, generate_cutouts

warnings.simplefilter(action="ignore", category=FutureWarning)


def main():

    parser = argparse.ArgumentParser(
        description="Train the CNN on different dataset sizes"
    )

    parser.add_argument(
        "--path-cube-validation",
        dest="path_cube_validation",
        required=True,
        type=str,
        help="Path to the datacube containing data used as validation set, including its name and extension.",
    )

    parser.add_argument(
        "--model-path",
        dest="path_model",
        required=True,
        type=str,
        help="Path where to the pretrained model, including it's name and extension",
    )

    parser.add_argument(
        "--threshold",
        dest="threshold",
        required=False,
        type=float,
        default=0.5,
        help="The threshold to define classes True and False" "(Default: 0.5)",
    )

    parser.add_argument(
        "--path-diagnostics",
        dest="path_diagnostics",
        required=False,
        type=str,
        default="cnn_results",
        help="path where to store the folders of the misclassified candidates",
    )

    args = parser.parse_args()
    diags = get_diagnostics(
        args.path_model,
        args.path_cube_validation,
        threshold=args.threshold,
    )

    print_diagnostics(diags)

    generate_cutouts(
        args.path_model,
        args.path_cube_validation,
        args.path_diagnostics,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
