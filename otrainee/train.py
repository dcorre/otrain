#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @file		train.py
# @brief	Train a CNN for detecting moving point-sources
# @date		01/11/2018
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# 	This file part of:	P9 search scripts
#
# 	Copyright:		(C) 2018 IAP/CNRS/SorbonneU
#
# 	Author:			Emmanuel Bertin (IAP)
#
# 	License:		GNU General Public License
#
# 	Bertinoscopic is free software: you can redistribute it and/or modify
# 	it under the terms of the GNU General Public License as published by
# 	the Free Software Foundation, either version 3 of the License, or
# 	(at your option) any later version.
# 	Bertinoscopic is distributed in the hope that it will be useful,
# 	but WITHOUT ANY WARRANTY; without even the implied warranty of
# 	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# 	GNU General Public License for more details.
# 	You should have received a copy of the GNU General Public License
# 	along with Bertinoscopic. If not, see <http://www.gnu.org/licenses/>.
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    Original script modified by: David Corre (IJCLab/CNRS)

import sys
import os
import errno
import shutil

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import argparse
from tensorflow.keras.utils import multi_gpu_model
from otrainee.utils import getpath, rm_p, mkdir_p

from otrainee.plot_results import plot_roc, plot_recall, plot_prob_distribution
from otrainee.diagnostics import get_diagnostics, generate_cutouts, print_diagnostics


def build_model(ima, nclass, dropout=0.3):
    """build model"""
    # define dropout percentageof each dropout
    dprob = np.array([dropout, dropout, dropout])
    # define padding
    padding = "same"  # valid, same
    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(
            16, (3, 3), activation="elu", padding=padding, input_shape=ima.shape[1:]
        )
    )
    model.add(
        keras.layers.Conv2D(
            32, (3, 3), activation="elu", padding=padding, input_shape=ima.shape[1:]
        )
    )
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(dprob[0]))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="elu", padding=padding))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(dprob[1]))
    model.add(keras.layers.Conv2D(128, (3, 3), activation="elu", padding=padding))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(dprob[1]))
    model.add(keras.layers.Conv2D(256, (3, 3), activation="elu", padding=padding))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(dprob[2]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="elu"))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dprob[2]))
    model.add(keras.layers.Dense(256, activation="elu"))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(nclass, activation="softmax"))
    return model


def train(
    path_cube,
    path_model,
    modelname,
    epochs=10,
    frac=0.15,
    dropout=0.3,
    threshold=0.53,
    dataset_size=None,
    nb_gpus=-1,
):
    """Train CNN with simulated data"""

    # dataset_size: is the training executed in the size optimisation loop or not
    # dataset_size = None if not
    # dataset_size = [size, randomize]

    path_model = os.path.join(path_model, "CNN_training", modelname)
    mkdir_p(path_model)

    # outputname for the trained model
    model_name = os.path.join(path_model, "%s.h5" % modelname)

    print("Loading " + path_cube + " ...", end="\r", flush=True)
    data = np.load(path_cube)
    ima = data["cube"]
    lab = keras.utils.to_categorical(data["labels"])
    mag = data["mags"]
    errmag = data["errmags"]
    band = data["filters"]
    candids = data["candids"]
    nclass = lab.shape[1]
    n = ima.shape[0]

    if dataset_size is None:
        size = n
        randomize = np.arange(n)
        np.random.shuffle(randomize)
    else:
        assert len(dataset_size) == 2
        size = dataset_size[0]
        randomize = dataset_size[1]

    # Number of data used for the validation test
    nt = int(size * frac)

    print("Shuffling data ...", end="\r", flush=True)
    ima = ima[randomize]
    lab = lab[randomize]
    mag = mag[randomize]
    errmag = errmag[randomize]
    band = band[randomize]
    candid = candids[randomize]
    nclass = lab.shape[1]

    print("Splitting dataset ...", end="\r", flush=True)
    imal = ima[nt:size]
    labl = lab[nt:size]
    magl = mag[nt:size]
    errmagl = errmag[nt:size]
    bandl = band[nt:size]
    candidl = candid[nt:size]

    imat = ima[:nt]
    labt = lab[:nt]
    magt = mag[:nt]
    errmagt = errmag[:nt]
    bandt = band[:nt]
    candidt = candid[:nt]

    # save datacube with data used for validation
    # useful for diagnostics
    outdir = os.path.join(path_model, "validation_set", "datacube")
    mkdir_p(outdir)
    npz_name = "cube_validation.npz"
    path_cube_test = os.path.join(outdir, npz_name)
    np.savez(
        path_cube_test,
        cube=imat,
        labels=labt,
        mags=magt,
        errmags=errmagt,
        candids=candidt,
    )

    # Build CNN model
    model = build_model(ima, nclass, dropout)

    # If GPUs are used
    if nb_gpus > 0:
        parallel_model = multi_gpu_model(model, gpus=nb_gpus)

        parallel_model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.001),
            # optimizer=keras.optimizers.Nadam(),
            metrics=["accuracy"],
        )

        parallel_model.fit(
            imal,
            labl,
            batch_size=1024,
            epochs=epochs,
            verbose=1,
            validation_data=(imat, labt),
        )

        score = parallel_model.evaluate(imat, labt, verbose=0)
        # save does not work on multi_gpu_model
        # parallel_model.save(model_name)
        labp = parallel_model.predict(imat)

    # If no GPU is used
    else:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.001),
            # optimizer=keras.optimizers.Nadam(),
            metrics=["accuracy"],
        )
        # log = keras.callbacks.ModelCheckpoint(
        #       'callbacks.h5', monitor='val_loss', verbose=0,
        #       save_best_only=True, save_weights_only=False,
        #       mode='auto', period=1)
        # log = keras.callbacks(TensorBoard(
        #        log_dir='./logs', histogram_freq=5, batch_size=1024,
        #        write_graph=True, write_grads=False, write_images=False,
        #        embeddings_freq=0, embeddings_layer_names=None,
        #        embeddings_metadata=None, embeddings_data=None,
        #        update_freq='epoch'))
        history = model.fit(
            imal,
            labl,
            batch_size=1024,
            epochs=epochs,
            verbose=1,
            validation_data=(imat, labt),
        )
        score = model.evaluate(imat, labt, verbose=0)
        labp = model.predict(imat)

    # Save trained model in hdf5 format
    model.save(model_name)

    # Run some diagnostics on the training
    diags = get_diagnostics(model_name, path_cube_test, threshold)

    # if this training is not run within the size_optimize loop, we can plot the following figures
    if dataset_size is None:
        _, axis = plt.subplots()
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(os.path.join(path_model, modelname + "_loss_vs_epochs.png"))

        _, axis = plt.subplots()
        axis.set_xlabel("epoch")
        axis.set_ylabel("accuracy")
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.legend(["train", "test"], loc="lower left")
        plt.savefig(os.path.join(path_model, modelname + "_accuracy_vs_epochs.png"))

        plot_roc(model_name, path_cube_test, path_model, 0.53)
        plot_recall(model_name, path_cube_test, path_model, 0.53)
        plot_prob_distribution(model_name, path_cube_test, path_model)
        print_diagnostics(diags)
        generate_cutouts(model_name, path_cube_test, path_model, 0.53)

    return history, diags
