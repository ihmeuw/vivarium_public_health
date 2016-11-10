Creating a development environment
==================================

Before we start you should setup a working environment on the cluster where you will have access to GBD data. To do that follow the instructions `here <https://hub.ihme.washington.edu/display/IHD/Create+virtual+environments+for+python+with+conda>`_. The only change is to say ``python=3.5`` where they say ``python=2.7`` because we're more optimistic than them. You can call your environment whatever you like but I'd suggest `ceam_development` because that's what we're up to.

Once the environment is created, activate it:

.. code-block:: console

    $ source activate ceam_development

You'll need to do that any time you start working on CEAM.

Then setup the directory where you plan to work. Despite what some of the training pages on the hub say, the sysadmins assure me that having git repositories in your home directory is fine so that's what I recommend.

.. code-block:: console

    $ cd ~/
    $ mkdir ceam_development
    $ cd ceam_development

You'll need the three ceam packages: ``ceam``, ``ceam_inputs`` and ``ceam_public_health``. Because you're going to be doing development an the whole system, you'll want to install the source for all three:

.. code-block:: console

    $ git clone https://stash.ihme.washington.edu/scm/cste/ceam.git
    $ git clone https://stash.ihme.washington.edu/scm/cste/ceam-inputs.git
    $ git clone https://stash.ihme.washington.edu/scm/cste/ceam-public-health.git

You should now have three directories. We want to install each one into our environment in development mode:

.. code-block:: console

    $ cd ceam
    $ python setup.py develop
    $ cd ../ceam-inputs
    $ python setup.py develop
    $ cd ../ceam-public-health
    $ python setup.py develop
