Tobacco smoking: effect of interventions
========================================

.. todo:: Describe the input data requirements.

.. todo:: Show the simulation definitions.

.. todo:: Show the effect of each intervention.

Tobacco eradication
-------------------

We add the ``TobaccoEradication`` component, and specify at what year it comes
into effect.
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

We add the ``TobaccoFreeGeneration`` component, and specify at what year it
comes into effect.
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

We enable the ``tobacco_tax`` option of the tobacco component.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   configuration:
       # Other configuration settings ...
       tobacco:
           tobacco_tax: True
