Tobacco smoking: effect of interventions
========================================

.. todo:: Show the simulation definitions.

.. todo:: Show the effect of each intervention.

Tobacco eradication
-------------------

.. py:currentmodule:: vivarium_public_health.mslt.intervention

We add the :class:`TobaccoEradication` component, and specify at what year it
comes into effect.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   components:
       vivarium_public_health:
           mslt:
               # Other components ...
               intervention:
                   TobaccoEradication()

   configuration:
       # Other configuration settings ...
       tobacco_eradication:
           year: 2011

Tobacco-free generation
-----------------------

We add the :class:`TobaccoFreeGeneration` component, and specify at what year
it comes into effect.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   components:
       vivarium_public_health:
           mslt:
               # Other components ...
               intervention:
                   TobaccoFreeGeneration()

   configuration:
       # Other configuration settings ...
       tobacco_free_generation:
           year: 2011

Tobacco tax
-----------

.. py:currentmodule:: vivarium_public_health.mslt.delay

We enable the ``tobacco_tax`` option of the tobacco risk factor
(:class:`DelayedRisk`).
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   configuration:
       # Other configuration settings ...
       tobacco:
           tobacco_tax: True
