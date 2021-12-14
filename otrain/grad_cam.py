#!/usr/bin/env python3
# -*- coding: utf-8 -*-import sys
#
# @file		grad_cam.py
# @brief	generate class activation maps for candidates after classification
# @date		17/08/2021

import os
import errno
import shutil
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from astropy.visualization import (
    LinearStretch,
    ImageNormalize,
    ZScaleInterval,
)


def get_img_array(array):
    # We add a dimension to transform our array into a "batch"
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_and_display_gradcam(i, img, heatmap, cam_path, title, alpha=0.5):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    heatmap_resized = resize(heatmap, (img.shape[0], img.shape[1]), anti_aliasing=True)

    plt.figure()
    norm = ImageNormalize(
        # subimage - np.median(subimage),
        img,
        interval=ZScaleInterval(),
        stretch=LinearStretch(),
    )
    # create a superimposed image
    plt.imshow(img[:, :, 0], cmap="gray", norm=norm)
    plt.imshow(heatmap_resized, cmap=jet, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.savefig(cam_path, fmt="png")


def get_data(path_model, path_cube_test, threshold=0.5, n_true=10):

    # load test data
    data = np.load(path_cube_test)
    imat = data["cube"]
    labt = data["labels"]
    magt = data["mags"]
    errmagt = data["errmags"]
    candidt = data["candids"]
    # load pretrained model
    model = keras.models.load_model(path_model)

    # get predictions and apply threshold
    labp = model.predict(imat)
    y_pred = np.asarray([int(x[1] > threshold) for i, x in enumerate(labp)])
    y_valid = np.argmax(labt, axis=1)
    list_tp = (y_valid == 1) & (y_pred == 1)
    list_tn = (y_valid == 0) & (y_pred == 0)
    list_fp = (y_valid == 0) & (y_pred == 1)
    list_fn = (y_valid == 1) & (y_pred == 0)
    indices_fp = [i for i, x in enumerate(list_fp) if x]
    indices_fn = [i for i, x in enumerate(list_fn) if x]
    # We choose randomly a sample of well classified candidates
    # the sample size = n_true
    indices_tp = random.sample([i for i, x in enumerate(list_tp) if x], n_true)
    indices_tn = random.sample([i for i, x in enumerate(list_tn) if x], n_true)
    return (
        imat,
        labp,
        magt,
        errmagt,
        candidt,
        indices_fp,
        indices_fn,
        indices_tn,
        indices_tp,
    )


def main_grad_cam(
    path_model, cam_path, path_cube_test, alpha=0.5, threshold=0.5, n_true=10
):
    """This function apply the grad-cam algorithm on few of the well classified
    transients, and all the misclassified ones to see which pixels triggered the classification"""

    # load model and data
    (
        imat,
        labp,
        magt,
        errmagt,
        candidt,
        indices_fp,
        indices_fn,
        indices_tn,
        indices_tp,
    ) = get_data(path_model, path_cube_test, threshold, n_true)
    model = keras.models.load_model(path_model)

    # get grad_cam output for Artefacts well classified
    outdir = "grad_cam/True Negative"
    os.makedirs(outdir, exist_ok=True)
    for i, x in enumerate(indices_tn):
        print(f"cutout_TN_{i}")
        print(f"prediction: {labp[x][1]:.3f} \n")
        img_array = get_img_array(imat[x])
        outname = os.path.join(outdir, f"cutout_CAM_{i}.png")
        heatmap = make_gradcam_heatmap(img_array, model, "conv2d_4", pred_index=0)
        save_and_display_gradcam(
            i,
            imat[x],
            heatmap,
            outname,
            "Cand: {0} Pred: {3:.3f} \nMag: {1:.2f} +/- {2:.2f} ".format(
                candidt[x], magt[x], errmagt[x], labp[x][1]
            ),
            alpha=alpha,
        )

    # get grad_cam output for true transients well classified
    outdir = "grad_cam/True Positive"
    os.makedirs(outdir, exist_ok=True)
    for i, x in enumerate(indices_tp):
        print(f"cutout_TP_{i}")
        print(f"prediction: {labp[x][1]:.3f} \n")
        img_array = get_img_array(imat[x])
        outname = os.path.join(outdir, f"cutout_CAM_{i}.png")
        heatmap = make_gradcam_heatmap(img_array, model, "conv2d_4", pred_index=1)
        save_and_display_gradcam(
            i,
            imat[x],
            heatmap,
            outname,
            "Cand: {0} Pred: {3:.3f} \nMag: {1:.2f} +/- {2:.2f} ".format(
                candidt[x], magt[x], errmagt[x], labp[x][1]
            ),
            alpha=alpha,
        )

    # get grad_cam output for Artefacts misclassified, to see why they were put in the true folder
    outdir = "grad_cam/False Positive"
    os.makedirs(outdir, exist_ok=True)
    for i, x in enumerate(indices_fp):
        print(f"cutout_FP_{i}")
        print(f"prediction: {labp[x][1]:.3f}\n")
        img_array = get_img_array(imat[x])
        outname = os.path.join(outdir, f"cutout_CAM_{i}.png")
        heatmap = make_gradcam_heatmap(img_array, model, "conv2d_4", pred_index=1)
        save_and_display_gradcam(
            i,
            imat[x],
            heatmap,
            outname,
            "Cand: {0} Pred: {3:.3f} \nMag: {1:.2f} +/- {2:.2f} ".format(
                candidt[x], magt[x], errmagt[x], labp[x][1]
            ),
            alpha=alpha,
        )

    # get grad_cam output for true transients misclassified to see why they were put in the False folder
    outdir = "grad_cam/False Negative"
    os.makedirs(outdir, exist_ok=True)
    for i, x in enumerate(indices_fn):
        print(f"cutout_FN_{i}")
        print(f"prediction: {labp[x][1]:.3f}\n")

        img_array = get_img_array(imat[x])
        outname = os.path.join(outdir, f"cutout_CAM_{i}.png")
        heatmap = make_gradcam_heatmap(img_array, model, "conv2d_4", pred_index=0)
        save_and_display_gradcam(
            i,
            imat[x],
            heatmap,
            outname,
            "Cand: {0} Pred: {3:.3f} \nMag: {1:.2f} +/- {2:.2f} ".format(
                candidt[x], magt[x], errmagt[x], labp[x][1]
            ),
            alpha=alpha,
        )
