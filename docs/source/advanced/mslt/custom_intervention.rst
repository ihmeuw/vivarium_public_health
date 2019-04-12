Writing a custom intervention
=============================

As :ref:`explained earlier <concept_intervention>`, an intervention will
typically affect the exposure distribution of a risk factor by modifying one
(or more) of:

* The rate(s) that affect the exposure (e.g., uptake of tobacco smoking);

* The prevalence of exposure categories (e.g., moving people from one exposure
  category to another); and

* The relative risk(s) associated with an exposure category.

.. note:: Interventions may also directly affect chronic and acute diseases,
   by modifying any of the rates associated with those diseases.
   This is very similar to modifying any of the rates that affect the exposure
   of a risk factor; the **only** difference is the choice of which rate(s)
   will be affected.

Structure of an intervention component
--------------------------------------

.. py:currentmodule:: vivarium_public_health.mslt.intervention

An intervention component will comprise the following methods:

* A constructor (``__init__``) that will **normally** accept two arguments:

  * The ``self`` parameter (a reference to the component *instance*); and

  * A ``name`` that will be used to identify this intervention, and which may
    be used in the configuration section of a simulation definition in order
    to define settings for this intervention.

* A ``setup(self, builder)`` method that will:

  * Load any required input value or rate tables.

  * Read any intervention-specific settings from ``builder.configuration``,
    such as the year at which the intervention will come into effect.

  * Register value and/or rate modifiers (if required).

  * Register a time-step event handler to, e.g., move people from one exposure
    category to another (if required).

* Some number of value and/or rate modifiers (if required).

* The time-step event handler (if required).

Example of an intervention component
------------------------------------

As an example, we will walk through each of these methods for the
:class:`TobaccoEradication` intervention.
This intervention is quite simple, because it doesn't need to load any input
data tables, and it has an all-or-nothing effect on a risk factor rate.

Recall that this intervention is controlled by a single configuration setting:

.. code-block:: yaml

   configuration:
       tobacco_eradication:
           year: 2011

The constructor
^^^^^^^^^^^^^^^

This intervention is currently hard-coded to modify the ``'tobacco'`` risk
factor, which it stores in ``self.exposure``.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/intervention.py
   :pyobject: TobaccoEradication.__init__

The setup method
^^^^^^^^^^^^^^^^

This method performs a several necessary house-keeping tasks:

* It retrieves the year at which the intervention comes into effect
  (specified in the configuration section, as shown above) and stores it in
  ``self.year``.

* It stores a reference to the simulation clock in ``self.clock``, so that it
  can detect when to come into effect.

* It registers a modifier for the ``tobacco_intervention.incidence`` rate
  (i.e., the uptake rate in the intervention scenario).

* It identifies which columns it needs to access in order to move people from
  the **currently smoking** exposure category to the **0 years
  post-cessation** exposure category.
  Note that these column names are defined by the risk factor component; in
  this case, see the documentation for the
  :meth:`~vivarium_public_health.mslt.delay.DelayedRisk.get_bin_names` method
  of the :class:`~vivarium_public_health.mslt.delay.DelayedRisk` component.

* It then informs the framework that it will need access to these exposure
  columns, and stores this "view" in ``self.population_view``.

* Finally, it registers an event handler that will be called **before** each
  time-step (by selecting the "time_step__prepare" event) that will perform
  the movement of people between exposure categories.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/intervention.py
   :pyobject: TobaccoEradication.setup

The rate modifier
^^^^^^^^^^^^^^^^^

This method, which was registered as a modifier for the
``tobacco_intervention.incidence`` rate, will set the rate to zero once the
intervention is active.
Recall that ``self.year`` is the year at which this intervention comes into
effect.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/intervention.py
   :pyobject: TobaccoEradication.adjust_rate

.. note:: Once this intervention becomes active, this rate modifier applies an
   effect on every time-step.

The prevalence modifier
^^^^^^^^^^^^^^^^^^^^^^^

This method, which was registered to handle the "time_step__prepare" event,
checks if this is the year at which the intervention comes into effect (i.e.,
``self.year``) and, if so, it moves all of the people in the **currently
smoking** exposure category to the **0 years post-cessation** exposure
category.

There are three separate steps to make this change in exposure category:

1. Retrieve the current prevalence for the **currently smoking** and the **0
   years post-cessation** exposure categories, using the ``get`` method of
   ``self.population_view``;

2. Adjust the prevalence in each of these two exposure categories; and

3. Make these changes take effect in the underlying MSLT table, using the
   ``update`` method of ``self.population_view``.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/intervention.py
   :pyobject: TobaccoEradication.stop_current_use

.. note:: In contrast to the rate modifier method, this prevalence modifier
   only applies an effect for a single time-step (i.e., the year that the
   intervention comes into effect).
   In all other years, this method returns without having any effect.
