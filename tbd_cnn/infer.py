#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: David Corre, IAP, corre@iap.fr

"""

import sys

import errno
import glob
import os
import numpy as np
from astropy.io import fits
import keras
import argparse
from astropy.wcs import WCS
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table

from keras import activations
from tbd_cnn.utils import getpath, rm_p, mkdir_p
import matplotlib.pyplot as plt


def infer(path_cutouts, path_model, probratio):
    """Apply a previously trained CNN"""

    model_name = path_model
    probratio = 1.0 / float(probratio)
    sstart = int(0)

    # Get all the images
    filenames = sorted(glob.glob(os.path.join(path_cutouts, "*.fits")))
    print("Loading model " + model_name + " ...", end="\r", flush=True)

    model = keras.models.load_model(model_name)
    model.summary()

    cube = []
    newfilenames = []
    for ima in filenames:
        hdus = fits.open(ima, memmap=False)
        head = hdus[0].header
        # Discard cutout without the reauired size (on the image edges)
        if head["edge"] == "True":
            continue
        data = hdus[0].data
        cube.append(data)
        hdus.close()
        newfilenames.append(ima)

    # Convert lists to B.I.P. NumPy arrays
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim < 4:
        cube = np.reshape(cube, [cube.shape[0], cube.shape[1], cube.shape[2], 1])
    else:
        cube = np.moveaxis(cube, 1, -1)

    p = model.predict(cube)
    p2 = p[:, 1]

    # label[j] = p / (p + (1.0 - p) * probratio)

    RA_list = []
    Dec_list = []
    original_file = []
    Xpos_list = []
    Ypos_list = []
    Cand_ID = []
    mag = []
    magerr = []
    FWHM = []
    FWHM_PSF = []

    for i in range(len(p)):
        hdus = fits.open(newfilenames[i], memmap=False)
        head = hdus[0].header
        RA_list.append(head["RA"])
        Dec_list.append(head["DEC"])
        original_file.append(head["FILE"])
        Xpos_list.append(head["XPOS"])
        Ypos_list.append(head["YPOS"])
        Cand_ID.append(head["CANDID"])
        mag.append(head["MAG"])
        magerr.append(head["MAGERR"])
        FWHM.append(head["FWHM"])
        FWHM_PSF.append(head["FWHMPSF"])

    table = Table(
        [
            newfilenames,
            RA_list,
            Dec_list,
            original_file,
            Xpos_list,
            Ypos_list,
            mag,
            magerr,
            FWHM,
            FWHM_PSF,
            Cand_ID,
            p[:, 0],
            p[:, 1],
        ],
        names=[
            "filename",
            "RA",
            "Dec",
            "originalFile",
            "Xpos",
            "Ypos",
            "mag",
            "magerr",
            "FWHM",
            "FWHMPSF",
            "cand_ID",
            "label0",
            "label1",
        ],
    )
    table.write(
        os.path.join(path_cutouts, "infer_results.dat"),
        format="ascii.commented_header",
        overwrite=True,
    )
