#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Authors: David Corre, IAP, corre@iap.fr
         Kenza Makhlouf, Ecole Centrale Lille, kenza.makhlouf@centrale.centralelille.fr
"""

import argparse
import warnings

from otrainee.grad_cam import main_grad_cam

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
        "--cam_path",
        dest="cam_path",
        required=False,
        type=str,
        help="Path to the cam images",
    )

    parser.add_argument(
        "--n_random",
        dest="frac",
        required=False,
        type=int,
        default=30,
        help="The number of randomly chosen well classified canddates",
    )

    parser.add_argument(
        "--threshold",
        dest="threshold",
        required=False,
        type=float,
        default=0.5,
        help="threshold for classification",
    )

    args = parser.parse_args()
    main_grad_cam(
        args.path_model,
        args.cam_path,
        args.path_cubename,
        threshold=args.threshold,
        n_true=args.frac
    )


if __name__ == "__main__":
    main()
