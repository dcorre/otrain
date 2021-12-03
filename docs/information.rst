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


Important information
---------------------

Cutouts header
^^^^^^^^^^^^^^

The following keywords are mandatory in the cutouts header used for the training and inference:
* CAND_ID
* MAG
* MAGERR
* others?



Description of parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

**otrain-convert**

* ``--path-datacube``: path where the datacube is stored.
* ``--cube``: Name of the datacube (without its extension).
* ``--cutouts``: Path where the cutouts used for the training are stored.
* ``--frac-true``: Fraction of true candidates to consider in the training set. For instance 0.5 means 50% of true and 50% of false candidates. If you provide 5000 false candidates and 1000 true candidates and set this parameter to 0.5, only 1000 false candidates will be used (randomly chosen). (optional, defaut: 0.5)


**otrain-train**

**otrain-infer**

**otrain-checkinfer**

**otrain-diagnostics**

**otrain-plots-results**

**otrain-optimise-dataset-size**
