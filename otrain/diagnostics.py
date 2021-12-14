#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @file		diagnostics.py
# @brief	get diagnostics
# @date		16/06/2021

import os
import shutil
import random
import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)
from otrain.utils import make_figure


def create_folder(path):
    """creates a folder, if it already exists all contents will be deleted"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def get_results(path_model, path_cube_test):
    """load the model and get prediction results"""

    data = np.load(path_cube_test)
    imat = data["cube"]
    magt = data["mags"]
    errmagt = data["errmags"]
    candidt = data["candids"]
    # test labels are already saved as categorical matrix
    labt = data["labels"]

    model = keras.models.load_model(path_model)
    # get predictions
    labp = model.predict(imat)
    return (labp, labt, imat, magt, errmagt, candidt)


def get_diagnostics(path_model, path_cube_test, threshold=0.5):
    """returns all evaluation metrics"""
    labp, labt = get_results(path_model, path_cube_test)[:2]
    y_pred = [int(x[1] > threshold) for i, x in enumerate(labp)]
    y_valid = np.argmax(labt, axis=1)

    diags = (
        average_precision_score(labt, labp, average=None),
        precision_score(y_valid, y_pred),
        recall_score(y_valid, y_pred),
        f1_score(y_valid, y_pred),
        confusion_matrix(y_valid, y_pred),
        matthews_corrcoef(y_valid, y_pred),
    )
    return diags


def print_diagnostics(diags):
    print("\n")
    print(f"average accuracy score: {diags[0]}")
    print("Precision: %1.3f" % diags[1])
    print("Recall: %1.3f" % diags[2])
    print("F1 score: %1.3f" % diags[3])
    print("MCC score: %1.3f" % diags[5])
    print("Confusion matrix:")
    print(diags[4])
    print("\n")


def generate_cutouts(
    path_model, path_cube_test, dest_path, threshold=0.5, nb_samples=30
):

    labp, labt, imat, magt, errmagt, candidt = get_results(path_model, path_cube_test)
    y_pred = [int(x[1] > threshold) for i, x in enumerate(labp)]
    y_pred = np.asarray(y_pred)
    y_valid = np.argmax(labt, axis=1)

    folder_f = os.path.join(dest_path, "validation_set", "misclassified")
    outdirfp = os.path.join(folder_f, "images_FP")
    create_folder(outdirfp)

    outdirfn = os.path.join(folder_f, "images_FN")
    create_folder(outdirfn)

    list_f = y_valid != y_pred
    indices_f = [i for i, x in enumerate(list_f) if x]
    images_f = np.zeros((len(indices_f), imat.shape[1], imat.shape[1], imat.shape[3]))
    for i, index in enumerate(indices_f):
        images_f[i] = imat[index]
    print("creating cutouts of misclassified candidates")
    for i, dat in enumerate(images_f):
        if y_valid[indices_f[i]] == 0:
            outname = os.path.join(outdirfp, f"cutout_{i}.png")
            make_figure(
                dat[:, :, 0],
                outname,
                origin="upper",
                fmt="png",
                title="Cand: {0}\nMag: {1:.2f} +/- {2:.2f}".format(
                    candidt[indices_f[i]], magt[indices_f[i]], errmagt[indices_f[i]]
                ),
            )
        else:
            outname = os.path.join(outdirfn, f"cutout_{i}.png")
            make_figure(
                dat[:, :, 0],
                outname,
                origin="upper",
                fmt="png",
                title="Cand{0} \nMag: {1:.2f} +/- {2:.2f}".format(
                    candidt[indices_f[i]], magt[indices_f[i]], errmagt[indices_f[i]]
                ),
            )

    # Creating cutouts of 30 randomly chosen well classified candidates
    folder = os.path.join(dest_path, "validation_set", "well_classified")
    outdirtp = os.path.join(folder, "images_TP")
    create_folder(outdirtp)

    outdirtn = os.path.join(folder, "images_TN")
    create_folder(outdirtn)

    list_tp = (y_valid == 1) & (y_pred == 1)
    list_tn = (y_valid == 0) & (y_pred == 0)
    indices_tp = [i for i, x in enumerate(list_tp) if x]
    indices_tn = [i for i, x in enumerate(list_tn) if x]

    nb_samples_tp = min(nb_samples, len(indices_tp))
    nb_samples_tn = min(nb_samples, len(indices_tn))
    rand_tp = random.sample(range(0, len(indices_tp)), nb_samples_tp)
    rand_tn = random.sample(range(0, len(indices_tn)), nb_samples_tn)

    # take 30 random images of True candidates
    images_tp = np.zeros((nb_samples_tp, imat.shape[1], imat.shape[1], imat.shape[3]))
    for i, index in enumerate(rand_tp):
        images_tp[i] = imat[indices_tp[index]]

    print(f"generating {nb_samples_tp} cutouts of well classified True candidates")

    for i, dat in enumerate(images_tp):
        outname = os.path.join(outdirtp, f"cutout_{i}.png")
        make_figure(
            dat[:, :, 0],
            outname,
            origin="upper",
            fmt="png",
            title="Cand: {0}\nMag: {1:.2f} +/- {2:.2f}".format(
                candidt[indices_tp[i]], magt[indices_tp[i]], errmagt[indices_tp[i]]
            ),
        )

    # take the 30 images of False candidates
    images_tn = np.zeros((nb_samples_tn, imat.shape[1], imat.shape[1], imat.shape[3]))
    for i, index in enumerate(rand_tn):
        images_tn[i] = imat[indices_tn[index]]

    print(f"generating {nb_samples_tn} cutouts of well classified False candidates")

    for i, dat in enumerate(images_tn):
        outname = os.path.join(outdirtn, f"cutout_{i}.png")
        make_figure(
            dat[:, :, 0],
            outname,
            origin="upper",
            fmt="png",
            title="Cand: {0}\nMag: {1:.2f} +/- {2:.2f}".format(
                candidt[indices_tn[i]], magt[indices_tn[i]], errmagt[indices_tn[i]]
            ),
        )
