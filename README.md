# O'TRAINEE: Optical TRAnsient Identification NEtwork: Easy!

* Documentation: https://otrainee.readthedocs.io.

Development status
--------------------

[![Build Status](https://travis-ci.com/dcorre/otrainee.svg?branch=master)](https://travis-ci.com/dcorre/otrainee)
[![codecov](https://codecov.io/gh/dcorre/otrainee/branch/master/graphs/badge.svg)](https://codecov.io/gh/dcorre/otrainee/branch/master)
[![Documentation Status](https://readthedocs.org/projects/otrainee/badge/?version=latest)](https://otrainee.readthedocs.io/en/latest/?badge=latest)


Generic Tools to help identification of transients in astronomical images, based on Convolutional Neural Network.

Aim
---

* Implement a general tool based on machine learning to help identifying transient in astonmical images. 
* Effortless execution through the command line.
* Usable on different kind of images from different telescopes.
* OS-independent, so that it can work on Windows, Mac OS and Linux machines.



Features
--------

* Usage of Docker to allow deployment on different OS, container based on Ubuntu 20.04.   
* User provide image cutouts containing real and false transients.   
* These cutouts are used to train a CNN algorithm implemented with [Keras](https://keras.io/).   
* Built-in diagnostics help to characterise accuracy of training.   
* Usage of trained model to classify any new cutouts


Installation
------------

See documentation: https://otrainee.readthedocs.io.


Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
