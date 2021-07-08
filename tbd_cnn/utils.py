#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: David Corre, IAP, corre@iap.fr

"""

import errno
import glob
import os
import shutil
import subprocess
import sys
import importlib
import time
import gc
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy.table import Table
from astropy import wcs
from astropy.wcs import WCS
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import (
    MinMaxInterval,
    SqrtStretch,
    LogStretch,
    SinhStretch,
    LinearStretch,
    ImageNormalize,
    ZScaleInterval,
)
from copy import deepcopy


def cp_p(src, dest):
    try:
        shutil.copy(src, dest)
    except BaseException:
        pass


def mv_p(src, dest):
    try:
        shutil.move(src, dest)
    except BaseException:
        pass


def rm_p(src):
    fileList = glob.glob(src, recursive=False)
    for filePath in fileList:
        try:
            os.remove(filePath)
        except BaseException:
            pass


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def getpath():
    """Get the path to tbd_cnn module"""
    try:
        findspec = importlib.util.find_spec("tbd_cnn")
        path = findspec.submodule_search_locations[0]
    except BaseException:
        print("path to tbd_cnn can not be found.")

    return path


def extract_subimage(
    data,
    header,
    w,
    OT_coords,
    coords_type,
    size,
    FoV,
):
    """Method that extracts a sub-image centerd on a given position"""

    # Get physical coordinates of OT
    if coords_type == "world":
        # Get physical coordinates
        c = coord.SkyCoord(
            OT_coords[0], OT_coords[1], unit=(u.deg, u.deg), frame="icrs"
        )
        world = np.array([[c.ra.deg, c.dec.deg]])
        pix1, pix2 = w.all_world2pix(world, 1)[0]
        pix = [pix2, pix1]
        pixref = OT_coords
    elif coords_type == "pix":
        pix = OT_coords
        # ra, dec = w.all_pix2world(np.array(pix), 0)
        ra, dec = w.all_pix2world(pix[0], pix[1], 0)
        pixref = [float(ra), float(dec)]

    if FoV > 0:
        # Get pixel size in degrees
        try:
            pixSize = abs(float(header["CDELT1"]))
        except BaseException:
            pixSize = abs(float(header["CD1_1"]))
        # Compute number of pixels to reach desired FoV in arcseconds
        size = [int(FoV / (pixSize * 3600)), int(FoV / (pixSize * 3600))]

    # Extract subimage from image starting from reference pixel
    x1 = int(pix[0]) - int(size[0] / 2)
    if x1 < 0:
        x1 = 0
    x2 = int(pix[0]) + int(size[0] / 2)
    y1 = int(pix[1]) - int(size[1] / 2)
    if y1 < 0:
        y1 = 0
    y2 = int(pix[1]) + int(size[1] / 2)
    subimage = data[x1:x2, y1:y2]

    # Highest declination on top
    ra1, dec1 = w.all_pix2world(pix[0], y1, 0)
    ra2, dec2 = w.all_pix2world(pix[0], y2, 0)
    if dec1 > dec2:
        origin = "upper"
    else:
        origin = "lower"

    return subimage, origin, size, pixref


def _make_sub_image(table, fname):
    """Method to be used for multiprocessing"""
    # for table, fname in zip(tables, fnames):
    # Load file
    hdul = fits.open(fname, memmap=False)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()
    w = WCS(header)
    for row in table:

        subimage, origin, size, pixref = extract_subimage(
            data,
            header,
            w,
            row["OT_coords"],
            row["coords_type"],
            row["sizes"],
            row["FoVs"],
        )

        if row["fmts"] == "fits":
            make_fits(
                subimage, row["output_names"], header, size, pixref, row["info_dicts"]
            )
        else:
            make_figure(
                subimage, row["output_names"], origin, row["fmts"], row["title"]
            )
        # del subimage
        # del w
        # gc.collect()


def make_sub_image(
    filenames,
    output_names,
    OT_coords,
    coords_type="world",
    sizes=[200, 200],
    FoVs=-1,
    info_dicts=None,
    title=None,
    fmts="fits",
    nb_threads=4,
):
    """
    Extract sub-image around OT coordinates for the given size.

    Parameters
    ----------
    filename : path to image, string
        The file to read, with its extension. For ex: '/home/image.fits.gz'
    OT_coords : OT coordinates, list
        Coordinates of the OT, for instance [129.23, 45.27]. This coordinates
        are used as the center of the sub-image to create.
    coords_type : string, optional
        Either 'world' or 'pix'. 'world' means that coordinates are ra, dec
        expected in degrees format. 'pix' is the physical pixel coordinate
        on the detector, for instance [1248,2057].
        Default: 'world'
    size : list, optional
        define the size in pixels of the new sub-image.
        Default: [200,200]
    FoV: float, optional
        define the FoV in arcsec for the subimage. If -1 then the size is
        defined by `size`
    Returns
    -------
    subimage: array, float
              array of the data sub-image
    origin: string
            orientation of the figure
    """
    # Ensure all inputs are 1D
    filenames = np.atleast_1d(filenames)
    OT_coords = np.atleast_1d(OT_coords)
    coords_type = np.atleast_1d(coords_type)
    sizes = np.atleast_1d(sizes)
    FoVs = np.atleast_1d(FoVs)
    output_names = np.atleast_1d(output_names)
    fmts = np.atleast_1d(fmts)

    # Otherwise there is a problem using zip()
    if OT_coords.shape == (2,):
        OT_coords = [OT_coords]
    if sizes.shape == (2,):
        sizes = [sizes]
    if info_dicts is None:
        info_dicts = [None] * len(filenames)
    else:
        info_dicts = np.atleast_1d(info_dicts)

    if title is None:
        title = [None] * len(filenames)
    else:
        title = np.atleast_1d(title)

    subimages = []
    headers = []
    size_list = []
    pixref = []
    origins = []

    table = Table(
        [
            filenames,
            output_names,
            OT_coords,
            coords_type,
            sizes,
            FoVs,
            info_dicts,
            title,
            fmts,
        ],
        names=[
            "filenames",
            "output_names",
            "OT_coords",
            "coords_type",
            "sizes",
            "FoVs",
            "info_dicts",
            "title",
            "fmts",
        ],
    )

    tables = []
    fnames = []
    for filename in table.group_by("filenames").groups.keys:
        mask = table["filenames"] == filename[0]
        fnames.append(filename[0])
        tables.append(table[mask])

    pool = mp.Pool(nb_threads)
    pool.starmap(_make_sub_image, np.array([tables, fnames]).T)
    pool.close()
    pool.join()


def make_fits(data, output_name, header, size, pixref, info_dict=None):
    """Create a fits file from a data array"""
    hdu = fits.PrimaryHDU()
    hdu.data = data.astype(np.float32)
    # FIXME: need to adapt header here !!! Likely not correct
    header["CRPIX1"] = int(size[0] / 2)
    header["CRPIX2"] = int(size[1] / 2)
    header["CRVAL1"] = pixref[0]
    header["CRVAL2"] = pixref[1]
    if info_dict is not None:
        # Add information regarding the transients
        for key, value in info_dict.items():
            # print (key, value)
            header[key] = value
        if data.shape[0] != size[0] or data.shape[1] != size[1]:
            header["edge"] = "True"
        else:
            header["edge"] = "False"
    hdu.header = header
    hdu.writeto(output_name, overwrite=True)
    del hdu
    gc.collect()


def make_figure(data, output_name, origin, fmt, title=None):
    """Make a figure from a data array"""

    norm = ImageNormalize(
        # subimage - np.median(subimage),
        data,
        interval=ZScaleInterval(),
        stretch=LinearStretch(),
    )

    plt.figure()
    plt.imshow(data, cmap="gray", origin=origin, norm=norm)
    # plt.imshow(norm(subimage),cmap="gray", origin=origin)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_name, format=fmt)
    plt.close()

    # Much faster but without annotations
    # plt.imsave(output_name, norm(subimage),
    #            cmap="gray", origin=origin, format=fmt)
    # plt.close()


def _combined_cutouts(
    filenames,
    OT_coords,
    coords_type,
    size,
    FoV,
):
    """Method that could be parallelised"""

    subimages = []
    origins = []
    for filename in filenames:
        hdul = fits.open(filename, memmap=False)
        data = hdul[0].data
        header = hdul[0].header
        hdul.close()
        w = WCS(header)

        subimage, origin, size, pixref = extract_subimage(
            data, header, w, OT_coords, coords_type, size, FoV
        )

        subimages.append(subimage)
        origins.append(origin)

    return subimages, origins


def combine_cutouts(
    filenames,
    OT_coords,
    coords_type="world",
    output_name="cutout_comb.png",
    size=[200, 200],
    FoV=-1,
    title=None,
    fmts="png",
):
    """Create a png file with the cutouts from science image,
    reference image and substarcted image."""

    # FIXME: need to optimise this function
    # May be provide the list of cutouts to create as inputs
    # to reuse the plt axes and avoid recreating a new pyplot
    # frame each time.

    datas, origins = _combined_cutouts(filenames, OT_coords, coords_type, size, FoV)

    norm1 = ImageNormalize(
        datas[0],  # - np.median(data1),
        interval=ZScaleInterval(),
        stretch=LinearStretch(),
    )
    norm2 = ImageNormalize(
        datas[1],  # - np.median(data2),
        interval=ZScaleInterval(),
        stretch=LinearStretch(),
    )
    norm3 = ImageNormalize(
        datas[2],  # - np.median(data3),
        interval=ZScaleInterval(),
        stretch=LinearStretch(),
    )

    # stretch=SinhStretch())
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Substracting by the median is a trick to highlight the source
    # when skybackground is important. It is not correct but just
    # used for illustration.
    # axs[1].imshow(data1 - np.median(data1),
    axs[1].imshow(datas[0], cmap="gray", origin=origins[0], norm=norm1)
    axs[1].set_xlabel("Science", size=20)
    # axs[2].imshow(data2 - np.median(data2),
    axs[2].imshow(datas[1], cmap="gray", origin=origins[1], norm=norm2)
    axs[2].set_xlabel("Reference", size=20)
    # axs[0].imshow(data3 #- np.median(data3),
    axs[0].imshow(datas[2], cmap="gray", origin=origins[2], norm=norm3)
    axs[0].set_xlabel("Residuals", size=20)
    if title is not None:
        fig.suptitle(title, size=20)
    # Tight_layout() does not support suptitle so need to do it manually.
    fig.tight_layout(rect=[0, 0.03, 1, 0.80])
    fig.savefig(output_name)
    plt.close()


def get_corner_coords(filename):
    """Get the image coordinates of an image"""

    header = fits.getheader(filename)
    # Get physical coordinates
    Naxis1 = header["NAXIS1"]
    Naxis2 = header["NAXIS2"]

    pix_coords = [[0, 0, Naxis1, Naxis1], [0, Naxis2, Naxis2, 0]]

    # Get physical coordinates
    w = WCS(header)
    ra, dec = w.all_pix2world(pix_coords[0], pix_coords[1], 1)

    return [ra, dec]
