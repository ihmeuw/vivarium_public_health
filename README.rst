======================
Vivarium Public Health
======================

**NOTE: This repository is archived and will receive no further updates.**

The ``vivarium_public_health`` package's development has migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **PyPI distribution:** ``vivarium-public-health`` (unchanged - same name)
- **Import path:** ``vivarium_public_health`` -> ``vivarium.public_health``
- **Source:** ``ihmeuw/vivarium_public_health`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/public-health/``)
- **Docs:** https://vivarium-public-health.readthedocs.io/

This repository's final release was ``v5.1.14``. The ``vivarium-public-health``
distribution name is now published from the monorepo starting at ``v6.0.0``, so
``pip install vivarium-public-health`` resolves to the monorepo release (which
imports as ``vivarium.public_health``). This repository is frozen and will not
receive updates.

To migrate fully to the new package
-----------------------------------

**Install (unchanged):**

.. code-block:: bash

    pip install vivarium-public-health

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
