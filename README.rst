======================
Vivarium Public Health
======================

**NOTE: This repository has been archived.**

The ``vivarium_public_health`` package has been renamed and migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **Import path:** ``vivarium_public_health`` -> ``vivarium.public_health``
- **Source:** ``ihmeuw/vivarium_public_health`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/public-health/``)

To migrate fully to the new package
-----------------------------------

**Install:**

.. code-block:: bash

    pip install vivarium-public-health  # no change here

**Import:**

.. code-block:: python

    import vivarium.public_health  # was: import vivarium_public_health

Original package overview
=========================

.. image:: https://badge.fury.io/py/vivarium-public-health.svg
    :target: https://badge.fury.io/py/vivarium-public-health

.. image:: https://github.com/ihmeuw/vivarium_public_health/actions/workflows/build.yml/badge.svg?branch=main
    :target: https://github.com/ihmeuw/vivarium_public_health
    :alt: Latest Version

.. image:: https://readthedocs.org/projects/vivarium_public_health/badge/?version=latest
    :target: https://vivarium_public_health.readthedocs.io/en/latest/?badge=latest
    :alt: Latest Docs

.. image:: https://zenodo.org/badge/141212278.svg
   :target: https://zenodo.org/badge/latestdoi/141212278

This library contains several components for for modelling diseases and their interventions.

You can install ``vivarium_public_health`` from PyPI with pip:

  ``> pip install vivarium_public_health``

or build it from source with

  ``> git clone https://github.com/ihmeuw/vivarium_public_health.git``

  ``> cd vivarium_public_health``

  ``> python setup.py install``


`Check out the docs! <https://vivarium.readthedocs.io/projects/vivarium-public-health/en/latest/>`_
---------------------------------------------------------------------------------------------------
