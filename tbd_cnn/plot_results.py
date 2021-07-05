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

    # load pretrained model
    model = keras.models.load_model(path_model)

    # get predictions
    labp = model.predict(imat)
    return magt, labp, labt


def plot_recall(path_model, path_cube_test, path_plot, threshold=0.53):
    """saves precision-recall plot"""
    magt, labp, labt = get_results(path_model, path_cube_test)
    y_valid = np.argmax(labt, axis=1)
    _, axis = plt.subplots()
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    mag_min = np.min(magt[magt != 99])

    # plot the precision-recall curve of each magnitude interval [lim_min,lim]
    lim_min = 0
    for lim in np.linspace(mag_min, 25, 6):
        labpm = labp[(lim_min <= magt) & (magt <= lim)]
        labtm = y_valid[(lim_min <= magt) & (magt <= lim)]
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
                recall, precision, label="{0:.2f} < mag < {1:.2f}".format(lim_min, lim)
            )
            # plt.scatter(recall[x], precision[x], marker='o', color='black', label='Best = %f' % thresholds[x])
        lim_min = lim
    legend = axis.legend(loc="lower left")
    plt.savefig(os.path.join(path_plot, "recall.png"))


def plot_roc(path_model, path_cube_test, path_plot, threshold=0.53):
    """plot ROC curve and save figure"""
    magt, labp, labt = get_results(path_model, path_cube_test)
    y_valid = np.argmax(labt, axis=1)

    _, axis = plt.subplots()
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    mag_min = np.min(magt[magt != 99])

    # plot the roc curve of each magnitude interval [lim_min,lim]
    lim_min = 0
    for lim in np.linspace(mag_min, 25, 6):
        labpm = labp[(lim_min <= magt) & (magt <= lim)]
        labtm = y_valid[(lim_min <= magt) & (magt <= lim)]
        if labtm != []:
            fpr, tpr, thresholds = roc_curve(labtm, labpm[:, 1])
            gmeans = np.sqrt(tpr * (1 - fpr))

            # get the index of the maximum value of gmean
            ix = np.argmax(gmeans)
            print(
                "Interval: [%.3f,%.3f], Best Threshold=%.3f, G-Mean=%.3f"
                % (lim_min, lim, thresholds[ix], gmeans[ix])
            )
            plt.plot(fpr, tpr, label="{0:.2f} < mag < {1:.2f}".format(lim_min, lim))
            # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best = %f' % thresholds[ix])
        lim_min = lim
    legend = axis.legend(loc="lower left")
    plt.savefig(os.path.join(path_plot, "ROC.png"))


def plot_prob_distribution(path_model, path_cube_test, path_plot):
    """plot probability distribution"""
    _, labp, _ = get_results(path_model, path_cube_test)

    plt.hist(labp, bins=15)
    plt.gca().set(title="Probability distribution", ylabel="Frequency")
    plt.legend(["False", "True"], loc="upper center")
    plt.savefig(os.path.join(path_plot, "prob_distribution.png"))
