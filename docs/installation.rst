.. highlight:: shell

============
Installation
============

There are two ways for installing the project:
     * install all the dependencies yourself.
     * use a Docker image to run the code.

The Docker image contains all the C dependencies and python packages, that can be sometimes very time consuming to install depending on your OS. So I recommend using the Docker image.
The code is using some of the astromatic softwares that can be difficult to run on Windows. So, it is strongly advised  to use the Docker image for Windows users.


Prerequisites
-------------

For both ways you will need to get the project first.

The sources for otrainee can be downloaded from the `Github repo`_.

* Either clone public repository:

  If git is not installed on your machine, see: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

  Then clone the project:

.. code-block:: console

    git clone git://github.com/dcorre/otrainee

Note that you will need to update the project regularly to check for updates. Ideally each time you want to use it, type the following command to search for updates:

.. code-block:: console

    git pull origin master


* Or download:

  Simply download this `tarball`_. Or through the console:

.. code-block:: console

    curl  -OL https://github.com/dcorre/otrainee/tarball/master

Cloning the project allows to retrieve easily future code updates. If you downloaded the project, you will need to download it again to retrieve future updates.


Installation with Docker
------------------------

The usage of a Docker allows to build an OS environment on your machine and thus avoid compatibility problems when running the code under Linux, Mac or Windows. If you have not Docker installed on your machine install it first.

* Install the Docker desktop for your OS: https://docs.docker.com/get-docker/

* To run Docker without appending sudo, type:

.. code-block:: console

   sudo groupadd docker
   sudo usermod -aG docker $USER

Log out and log back in so that your group membership is re-evaluated. For more information see https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user.

You can test that Docker is installed correctly and can be run without sudo:

.. code-block:: console

   docker run hello-world


Download the otrainee Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To retrieve the Docker image:

.. code-block:: console

   docker pull dcorre/otrainee

Check that it appears in the list of images:

.. code-block:: console

   docker images


Installation without Docker
---------------------------

I advise to create a virtual environment to avoid messing with different python libraries version that could be already installed on your computer and required for other projects.

Install conda: https://docs.conda.io/en/latest/miniconda.html

You can also install everything with pip if you prefer not to use conda.

Python 3 environment:
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    conda create -n otrainee python=3 numpy scipy matplotlib astropy h5py scikit-image


Activate the environment:
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    conda activate otrainee


Install other libraries
^^^^^^^^^^^^^^^^^^^^^^^

Once you have activated the environment, install the packages that are not available with conda using pip:

.. code-block:: console

    python3 -m pip install keras tensorflow opencv-python-headless


.. _Github repo: https://github.com/dcorre/otrainee
.. _tarball: https://github.com/dcorre/otrainee/tarball/master


Testing that it is working
--------------------------

Run Docker
^^^^^^^^^^

Run the Docker image in the background:

.. code-block:: console

   docker run --name otrainee -dit -v /path_to_your_data/:/home/newuser/data/ dcorre/otrainee

| This means that you run the docker image `dcorre/otrainee`, and give the name `otrainee` to the created container.
| `-d` runs the container in backgound.   
| `-i` gives the possibility to enter in the container to run commands interactively in a bash terminal.
| `-t` allocates a pseudo-TTY. 
| The -v option means that you mount a volume in the Docker pointing to a directory on your computer. This allows to exchange data between the Docker and your machine.
| The volume is pointing to the directory containing your images on your machine. You need to edit the path before the ``:``.

Once you have executed this command, you can run any command in the container by typing:

.. code-block:: console

   docker exec otrainee ls
   docker exec otrainee pwd
   
to make a `ls` or a `pwd` in the container named `otrainee`, or any other bash commands.

**In the following, if you are using a Docker image just prepend the command `docker exec otrainee` to run the given commands within the container instead of your machine.** 

The container is alive as long as you do not shut down your machine. It is important to know that you can not give the same name to two containers. So if for some reasons you need to remove the current container to start a new one, type:

.. code-block:: console

   docker rm otrainee

You can list the containers, active or not, on your machine with:

.. code-block:: console

   docker ps -a



Install otrainee
^^^^^^^^^^^^^^^^
-----------------------
Inside the Docker image
-----------------------

Already installed.

--------------------
Without Docker image
--------------------

.. code-block:: console

   python3 setup.py develop



Run otrainee on a test image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test if otrainee is running normally:

.. code-block:: console

   otrainee-train -h

It should return you the list of accepted arguments for this executable.
