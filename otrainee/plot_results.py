#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @file		plotresults.py
# @brief	plot precision-recall and ROC
# @date		16/06/2021

import sys
import os
import errno
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import roc_curve, precision_recall_curve


def get_results(path_model, path_cube_test):
    """get predictions results, and test data"""
    data = np.load(path_cube_test)
    imat = data["cube"]
    labt = data["labels"]
    magt = data["mags"]
    errmagt = data["errmags"]
    # load pretrained model
    model = keras.models.load_model(path_model)

    # get predictions
    labp = model.predict(imat)
    return errmagt, magt, labp, labt


def roc(path_plot, valt, labp, y_valid, val_name, val_max):
    """plot the ROC curve corresponding to a certain range of: magnitude, 
    or uncertainty in magnitude"""
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    val_min = np.min(valt[(valt != 99) & (valt != 0)])

    lim_min = 0
    for lim in np.linspace(val_min, val_max, 6):
        labpm = labp[(lim_min <= valt) & (valt <= lim)]
        labtm = y_valid[(lim_min <= valt) & (valt <= lim)]
        if labtm != []:
            fpr, tpr, thresholds = roc_curve(
                labtm, labpm[:, 1], drop_intermediate=False
            )
            gmeans = np.sqrt(tpr * (1 - fpr))
            # if abs(lim_min - 16.23)<0.1:
            # print(thresholds)
            # get the index of the maximum value of gmean
            ix = np.argmax(gmeans)
            print(
                "Interval: [%.3f,%.3f], Best Threshold=%.3f, G-Mean=%.3f"
                % (lim_min, lim, thresholds[ix], gmeans[ix])
            )
            plt.plot(
                fpr, tpr, label="{0:.2f} < {1} < {2:.2f}".format(lim_min, val_name, lim)
            )

            # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best = %f' % thresholds[ix])
        lim_min = lim
    # plt.plot([0, 0, 1], [0, 1, 1],color="k", linestyle="--", label="ideal")
    # plt.plot([0, 1], [0, 1],color="k", linestyle="-.", label = "no skill")

    plt.legend(loc="lower right")
    plt.grid(color="k", linestyle="--", linewidth=0.1)
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, val_name + "_ROC.png"))


def recall(path_plot, valt, labp, y_valid, val_name, val_max):
    """plot the precision-recall curve corresponding to a certain range: magnitude, 
    or uncertainty in magnitude"""
    plt.figure()
    plt.xlabel("recall")
    plt.ylabel("precision")
    val_min = np.min(valt[(valt != 99) & (valt != 0)])

    lim_min = 0
    for lim in np.linspace(val_min, val_max, 6):
        labpm = labp[(lim_min <= valt) & (valt <= lim)]
        labtm = y_valid[(lim_min <= valt) & (valt <= lim)]
        if labtm != []:
            precision, recall, thresholds = precision_recall_curve(labtm, labpm[:, 1])
            fscore = (2 * precision * recall) / (precision + recall)

            # get the index of the maximum value of the f-score
            x = np.argmax(fscore)
            print(
                "Interval : [%.3f,%.3f], Best Threshold=%.3f, F-Score=%.3f"
                % (lim_min, lim, thresholds[x], fscore[x])
            )
            plt.plot(
                recall,
                precision,
                label="{0:.2f} < {1} < {2:.2f}".format(lim_min, val_name, lim),
            )
            # plt.scatter(recall[x], precision[x], marker='o', color='black', label='Best = %f' % thresholds[x])
        lim_min = lim
    plt.legend(loc="lower left")
    # plt.axhline(y=0.5, color="k", linestyle="-.", label="no skill")
    plt.grid(color="k", linestyle="--", linewidth=0.1)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, val_name + "_recall.png"))

    plt.figure()


def plot_roc(
    path_model, path_cube_test, path_plot, threshold=0.53, mag_max=25, errmag_max=0.4
):
    """plot ROC curves and save figures"""
    errmagt, magt, labp, labt = get_results(path_model, path_cube_test)
    y_valid = np.argmax(labt, axis=1)

    # plot the roc curve of each uncertainty of magnitude interval [lim_min,lim]
    roc(path_plot, errmagt, labp, y_valid, "errmag", errmag_max)
    # plot the roc curve of each magnitude interval [lim_min,lim]
    roc(path_plot, magt, labp, y_valid, "mag", mag_max)


def plot_recall(
    path_model, path_cube_test, path_plot, threshold=0.53, mag_max=25, errmag_max=0.4
):
    """saves precision-recall plots"""
    errmagt, magt, labp, labt = get_results(path_model, path_cube_test)
    y_valid = np.argmax(labt, axis=1)

    # plot the precision-recall curve of each uncertainty of magnitude interval [lim_min,lim]
    recall(path_plot, errmagt, labp, y_valid, "errmag", errmag_max)
    # plot the precision-recall curve of each magnitude interval [lim_min,lim]
    recall(path_plot, magt, labp, y_valid, "mag", mag_max)


def plot_prob_distribution(path_model, path_cube_test, path_plot, threshold=0.53):
    """plot probability distribution"""
    _, _, labp, _ = get_results(path_model, path_cube_test)

    # Even linear bin steps
    bins = np.linspace(0, 1, 15)
    plt.figure()
    plt.hist(
        labp[:, 1][np.where(labp[:, 1] > threshold)], bins=bins, label="Real", alpha=0.5
    )
    plt.hist(
        labp[:, 1][np.where(labp[:, 1] < threshold)],
        bins=bins,
        label="Bogus",
        alpha=0.5,
    )

    plt.title("Probability distribution")
    plt.ylabel("Frequency")
    plt.xlabel("probability")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, "prob_distribution.png"))
