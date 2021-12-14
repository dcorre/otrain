=======
Outputs
=======

Short explanation of the different output files. The convention is that the original name of the image is kept and some keywords are appended to it or its extension.

* ``test_loss_vs_epochs.png``: Plot of the loss values as function of the training epochs. Plot made in otrain-train

* ``test_accuracy_vs_epochs.png``: Plot of the Accuracy values as function of the training epochs. Plot made in otrain-train

* ``model_name.h5``: CNN model file to be executed by otrain-infer

* ``prob_distribution.png``: Plot the probability distribution of the cutouts used for the training. Plot made in otrain-plot-results

* ``model_F1_MCC.png``: Plot the f1-mcc curve and save figure in path_plot. Plot made in otrain-plot-results

* ``mag_ROC.png``: Plot the ROC curve corresponding to a certain range of magnitude. Plot made in otrain-plot-results

* ``errmag_ROC.png``: Plot the ROC curve corresponding to a certain range of magnitude uncertainties. Plot made in otrain-plot-results
 
* ``mag_recall.png``: Plot the precision-recall curve corresponding to a certain range of magnitude. Plot made in otrain-plot-results

* ``errmag_recall.png``: Plot the precision-recall curve corresponding to a certain range of magnitude uncertainties. Plot made in otrain-plot-results

* ``well_classified``: Folder storing 30 cutouts of well-classified candidates. folder created by otrain-train

* ``misclassified``: Folder storing 30 cutouts of misclassified candidates. folder created by otrain-train   

* ``cube_validation.npz``: datacube of cutouts used to validate the performance of the CNN. folder created by otrain-train
