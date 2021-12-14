===========================
Global description of usage
===========================


Objective
---------

The image substraction results in many false candidates due to bad astrometric calibration, bad kernel estimation, bad photometric calibration, etc...


These artefacts are usually easy to discard by eye, but it is time consuming and obviously not possible in an automatic process. So one soluton is to make use of machine learning to filter these false candidates, using for instance a CNN algorithm.



Process
-------

Launch the Doker image
^^^^^^^^^^^^^^^^^^^^^^



If you are using the Docker image, remember to launch once the container:

.. code-block:: console

   docker run --name otrain -dit -v /path_to_your_data/:/home/newuser/data/  dcorre/otrain

Replace:


* ``/path_to_your_data/`` with the path on your machine pointing to the data you want to analyse.


Then you only need to prepend `docker exec otrain` to the commands given below to execute them within the container instead of your machine.


Classify true and false candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The idea is to create 2 folders, one for the true candidates and one for the false candidates. You can classify them by eye, perform a crossmatch with variable stars catalogs, etc...
The main thing is to put what you consider true and false candidates in the respective folders.


Convert the cutouts into a single datacube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have classified your candidates, the next step is to trained the CNN algorithm to classify candidates. Before starting the training, we need to create a .npz datacube containing the candidates in the right format.

.. code-block:: console

    otrain-convert --path PATH_DATACUBE --cube CUBENAME --cutouts PATH_CUTOUTS --frac_true FRAC_TRUE --cutout-size CUTOUT_SIZE

Replace:

* ``PATH_DATACUBE`` with the path where you want to store your datacube.
* ``CUBENAME`` with the name of the datacube that will be created.
* ``PATH_CUTOUTS`` with the path to the folder containing the ``true`` and ``false`` folders.
* ``FRAC_TRUE`` with the fraction of the True events you want in the final datacube.
* ``CUTOUT_SIZE`` with the size of the cutouts (CUTOUT_SIZExCUTOUT_SIZE pixels) you want in the final datacube.

For example:

.. code-block:: console

    otrain-convert --path datacube_test --cube cube --cutouts candidates_training --frac_true 0.5 --cutout-size -1

The cutouts are taken from the ``False`` and ``True`` folders in ``candidates_training/`` and the cube will be created in ``datacube_test/cube.npz with the original size of the cutouts``


Train a model
^^^^^^^^^^^^^

Then you can start the training:

.. code-block:: console

    otrain-train --cube PATH_CUBENAME --model-path PATH_MODEL --model-name MODELNAME --epochs EPOCHS

Replace:

* ``PATH_CUBENAME`` with the path containing the datacube, including the filename and .npz extension.
* ``PATH_MODEL`` with the path where you want to store the trained model.
* ``MODELNAME`` with the name of the model that will be created.
* ``EPOCHS`` with the number of epochs to be launched for the training.

For example:

.. code-block:: console

    otrain-train --cube datacube_test/cube.npz --model-path model --model-name test --epochs 30

The model will be stored in ``model/CNN_training/test.h5 after 30 epochs of training``


Apply a trained model on candidates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It assumes that you already ran ``otrain`` on a set of images. For instance you have a ``candidates/`` folder containing the cutouts that need to be classified by the CNN algorithm. 


.. code-block:: console

    otrain-infer --cutouts PATH_CUTOUTS --model PATH_MODEL

Replace:

* ``PATH_CUTOUTS`` with the path containing the candidates cutouts.
* ``PATH_MODEL`` with the path to the trained CNN model, including its filnemame and .h5 extension.

For example:

.. code-block:: console

    otrain-infer --cutouts candidates --model model/CNN_training/test.h5

It will result a file ``infer_results.dat`` in the directory defined with ``--cutouts``, containing the probability that a source is a false (column: label0) or true (column: label1) transient.    
You can then apply a threshold on these probability to keep only some candidates. 

To visualize how these probabilities evolve with some of the candidates parameters (magnitude, FWHM) of your sample, you can use ``otrain-checkinfer``.

.. code-block:: console

    otrain-checkinfer --plots PATH_PLOTS --crossmatch PATH_CROSSMATCH --infer PATH_INFER

Replace:

* ``PATH_PLOTS`` with the path where you want to store the plots.
* ``PATH_CROSSMATCH`` with the path where the ``crossmatch.dat`` is stored.
* ``PATH_INFER`` with the path where the ``infer_results.dat`` is stored.


Type ``otrain-cnn_checkinfer -h`` to see the other optional arguments.

For example:

.. code-block:: console

    otrain-checkinfer --plots otrain_plots --crossmatch .  --infer candidates

It will results a folder ``CheckInfer`` containing some plots illustrating the dependence of the probability that a candidate is a true transient (returned by the CNN algorithm) as a function of magnitude and FWHM ratio (so far, can include more check in the future). It also compares this evolution for the simulated soures with respect to the non-simulated sources. It is also useful to get an idea of the FWHM ratio range that can be applied to filter the candidates.

General notes
^^^^^^^^^^^^^

You should have a similar number of true and false transients in your training sample. 

Ideally the training should be done on a few tens of images with taken in different observing conditions (elevation, seeing, moon phase, etc...) so that you can train a model that is representative enough of the images you can have, and thus not having to train a model for each sample of images you want to analyse.

Of course, if the computational time is not a constraint for you, it will be more accurate to perform a training on the images you want to analyse only, if you have a sufficient number of them.

Regarding the total number of transients required for an accurate training, you can start with a large number of cutouts and can use ``otrain-optimise-dataset-size`` to find out the minimum acceptable size. 

