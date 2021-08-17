#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import warnings
import os
import shutil

from otrainee.plot_results import plot_roc, plot_recall, plot_prob_distribution

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
        help="Path to the pretrained model, including it's name and extension",
    )

    parser.add_argument(
        "--path-plots",
        dest="path_plots",
        required=True,
        type=str,
        help="Path where to store the plots",
    )

    parser.add_argument(
        "--threshold",
        dest="threshold",
        required=False,
        type=float,
        default=0.5,
        help="The threshold to define classes True and False" "(Default: 0.5)",
    )

    args = parser.parse_args()

    # create the folder for the plots
    if os.path.exists(args.path_plots):
        shutil.rmtree(args.path_plots)
    os.makedirs(args.path_plots, exist_ok=True)

    plot_roc(
        args.path_model,
        args.path_cube_validation,
        args.path_plots,
        threshold=args.threshold,
    )

    plot_recall(
        args.path_model,
        args.path_cube_validation,
        args.path_plots,
        threshold=args.threshold,
    )

    plot_prob_distribution(args.path_model, args.path_cube_validation, args.path_plots)


if __name__ == "__main__":
    main()
