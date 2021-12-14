===================
General information
===================


Description of the process
--------------------------


Executables
^^^^^^^^^^^

All the scripts can be run with executable beginning with "gamdet-". Then you can provide the options through arguments from the command line, for instance type ``otrain-train -h`` to know the expected arguments for the executable performing astrometry calibration. Below the list of executables (use the ``-h`` options to see the expected arguments):

Main executables:

* **otrain-convert**: Convert the candidates cutouts (classified as true/false events) into a single datacube with the format expected by the CNN algorithm.

* **otrain-train**: train the CNN algorithm on this datacube.

* **otrain-infer**: Apply a trained CNN model on a set of candidates cutouts to assign a probability of being a true or false event regarding what has been used for the training.

* **otrain-checkinfer**: Do some plots to visualise the CNN training. Useful for chosing the probability threshold that will be used to classify a candidate as true or false event.

* **otrain-diagnostics**: Run diagnostics to estimate the quality of the training.

* **otrain-plots-results**: Plot some results associated to the diagnostics.

* **otrain-optimise-dataset-size**: Perform a training with different dataset size to study the performance of the training as a nummber of cutouts. Start from current size and run training on decreasing dataset sizes.

* **otrain-grad-cam**: Display the region of the cutouts where the CNN model focuses on to deliver its classification decision.

Important information
---------------------

Cutouts header
^^^^^^^^^^^^^^

The following keywords are mandatory in the cutouts header used for the training and inference:

* CANDID: name or alias of the transient candidate

* MAG: magnitude of the source

* MAGERR: error on the magnitude of the source

* FILTER: filter used for the observation

* FWHM: estimation of the Full-Width at Half Maximum of the source

* FILTER: filter used for the observation

* EDGE: True if the source is close an image edge (depends on the detection pipeline setup). False otherwise.


Description of parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

**otrain-convert**

* ``--path-datacube``: path where the datacube is stored.
* ``--cube``: Name of the datacube (without its extension).
* ``--cutouts``: Path where the cutouts used for the training are stored.
* ``--frac-true``: Fraction of true candidates to consider in the training set. For instance 0.5 means 50% of true and 50% of false candidates. If you provide 5000 false candidates and 1000 true candidates and set this parameter to 0.5, only 1000 false candidates will be used (randomly chosen). (optional, defaut: 0.5)
* ``--cutout-size``: Size of the cutout in pixels, NxN pixels. If you want to keep the full size of your cutouts, leave the optional value otherwise it will crop the cutout at the selected size keeping the candidate at the center of the image. (optional, defaut: -1)


**otrain-train**

* ``--cube``: Path to the datacube, including its name and extension.
* ``--model-path``: Path where to store the trained model.
* ``--model-name``: Name of the trained model.
* ``--epochs``: Number of epochs. (Default: 10)
* ``--frac``: Fraction of the data used for the validation sample. (Default: 0.15)
* ``--dropout``: Fraction used for each dropout. (Default: 0.3)
* ``--threshold``: Probability threshold used for diagnostics between 0 and 1. (Default: 0.5)

**otrain-infer**

* ``--cutouts``: Path to cutouts.
* ``--prob-ratio``: Proba ratio, not used at the moment.
* ``--model``: Path to the trained model, including its name and extension.

**otrain-checkinfer**

* ``--plots``: Path to folder where to store plots.
* ``--crossmatch``: Path to crossmatch.dat file.
* ``--infer``: Path to infer.dat file.
* ``--maglim``: Magnitudes used to split the magnitude range in the plots. (Default: 12 18 21)
* ``--cnn-problim``: CNN proba used to split the results in the plots. (Default: 0.1 0.5 0.7)
* ``--fwhm-ratio-lower``: Lower bound for the ratio FWHM / FWHM_PSF used for the plots. (Default: 0.5)
* ``--fwhm-ratio-upper``: Upper bound for the ratio FWHM / FWHM_PSF used for the plots. (Default: 4.2)

**otrain-diagnostics**

* ``--path-cube-validation``: Path to the datacube containing data used as validation set, including its name and extension.
* ``--model-path``: Path where to the pretrained model, including it's name and extension
* ``--threshold``: Probability threshold used for diagnostics between 0 and 1. (Default: 0.5)
* ``--path-diagnostics``: path where to store the folders of the misclassified candidates

**otrain-plot-results**

* ``--path-cube-validation``: Path to the datacube containing data used as validation set, including its name and extension.
* ``--model-path``: Path to the pretrained model, including it's name and extension
* ``--path-plots``: Path where to store the plots
* ``--threshold``: Probability threshold used for diagnostics between 0 and 1. (Default: 0.5)

**otrain-optimise-dataset-size**

* ``--cube``: Path to the datacube, including its name and extension.
* ``--model-path``: Path where to store the trained model.
* ``--model-name``: Name of the trained model.
* ``--epochs``: Number of epochs. (Default: 10)
* ``--n_sections``: The number of sections we want to divide the dataset into.(Default: 20)
* ``--dropout``: Fraction used for each dropout. (Default: 0.3)
* ``--outdir``: Path where to store the results.

**otrain-grad-cam**

* ``--cube``: Path to the datacube, including its name and extension.
* ``--model-path``: Path where to store the trained model.
* ``--cam_path``: Path to the cam images
* ``--n_random``: The number of randomly chosen well classified canddates
* ``--threshold``: Probability threshold used for diagnostics between 0 and 1. (Default: 0.5)

