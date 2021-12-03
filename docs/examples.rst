========
Examples
========

Launch the Doker image
----------------------

If you are using the Docker image, remember to launch once the container:

.. code-block:: console

   docker run --name otrain -dit -v /path_to_your_data/:/home/newuser/data/  dcorre/otrain

Replace:


* ``/path_to_your_data/`` with the path on your machine pointing to the data you want to analyse.


Then you only need to prepend `docker exec otrain` to the commands given below to execute them within the container instead of your machine.


TO DO
