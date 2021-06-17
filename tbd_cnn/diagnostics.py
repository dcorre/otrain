#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @file		diagnostics.py
# @brief	get diagnostics
# @date		16/06/2021

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


def get_results(path_model, path_cube_test):
    """load the model and get prediction results"""

    data = np.load(path_cube_test)
    imat = data["cube"]

    # test labels are already saved as categorical matrix
    labt = data["labels"]

    model = keras.models.load_model(path_model)
    # get predictions
    labp = model.predict(imat)
    return labp, labt


def get_diagnostics(path_model, path_cube_test, threshold):
    """returns all evaluation metrics"""
    labp, labt = get_results(path_model, path_cube_test)
    y_pred = [int(x[1] > threshold) for i, x in enumerate(labp)]
    y_valid = np.argmax(labt, axis=1)
    return (
        average_precision_score(labt, labp, average=None),
        precision_score(y_valid, y_pred),
        recall_score(y_valid, y_pred),
        f1_score(y_valid, y_pred),
        confusion_matrix(y_valid, y_pred),
        matthews_corrcoef(y_valid, y_pred),
    )
