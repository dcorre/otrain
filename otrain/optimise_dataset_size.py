#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @file		size_optimize.py
# @brief	get the dataset size above which training results stabilize
# @date		17/06/2021

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from otrain.train import train
from otrain.utils import mkdir_p


def optimise_dataset_size(
    path_cube,
    path_model,
    modelname,
    epochs,
    n_sections=21,
    dropout=0.3,
    outdir="optimise_dataset_dir",
):
    """trains the model on different sizes and plots the evolution of results vs dataset sizes"""

    mkdir_p(outdir)

    data = np.load(path_cube)
    ima = data["cube"]
    # get the datacube size
    n = ima.shape[0]

    # initialize lists
    sizes = []
    mcc = []
    acc = []
    fscores = []
    delta_acc = []
    losses = []
    # we shuffle only once
    randomize = np.arange(n)
    np.random.shuffle(randomize)

    for i in range(1, n_sections):
        size = int(((n * i) / (n_sections - 1)))

        # train function return history of the training,
        # and the metrics from get_diagnostics in diagnostics.py
        modelname_iteration = modelname + "_size_" + str(size)
        history, scores = train(
            path_cube,
            path_model,
            modelname_iteration,
            epochs,
            dataset_size=[size, randomize],
        )

        mcc.append(scores[5])
        fscores.append(scores[3])
        acc.append(history.history["val_accuracy"][-1])
        losses.append(history.history["val_loss"][-1])
        delta_acc.append(
            history.history["accuracy"][-1] - history.history["val_accuracy"][-1]
        )
        sizes.append(size)

    _, axis = plt.subplots()
    axis.set_xlabel("dataset size")
    axis.set_ylabel("scores")
    plt.plot(sizes, mcc, label="MCC")

    plt.plot(sizes, fscores, label="F1")

    plt.plot(sizes, acc, label="Accuracy")
    legend = axis.legend(loc="lower left")
    plt.savefig(os.path.join(outdir, modelname + "_scores_dataset_size.png"))

    _, axis = plt.subplots()
    axis.set_xlabel("dataset size")
    axis.set_ylabel("delta accuracy")
    plt.plot(sizes, delta_acc)
    plt.savefig(os.path.join(outdir, modelname + "_ecart_accuracy_size.png"))

    _, axis = plt.subplots()
    axis.set_xlabel("dataset size")
    axis.set_ylabel("loss")
    plt.plot(sizes, losses)
    plt.savefig(os.path.join(outdir, modelname + "_loss_dataset_size.png"))
