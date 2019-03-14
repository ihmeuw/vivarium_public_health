Writing a custom risk factor
============================

As :ref:`explained earlier <concept_risk_factor>`, a risk factor will
typically define a number of exposure categories, and each category will be
assigned one or more relative risks (e.g., for chronic disease incidence).
The primary concerns when writing a custom risk factor are:

* Identifying appropriate exposure categories;

* Identifying which rates will be modified by these exposure categories;

* Defining the relative risks for each rate, for each exposure category; and

* Defining transition rates between exposure categories (if applicable).

Once these concerns have been addressed, the input data requirements can be
identified, and input data tables can be prepared.

Structure of a risk factor component
------------------------------------

.. py:currentmodule:: vivarium_public_health.mslt.delay

A risk factor component will comprise the following methods:

* A constructor (``__init__``) that will **normally** accept two arguments:

  * The ``self`` parameter (a reference to the component *instance*); and

  * A ``name`` that will be used to identify this risk factor, and which may
    be used in the configuration section of a simulation definition in order
    to define settings for this risk factor.

* A ``setup(self, builder)`` method that will:

  * Load any required input value or rate tables, including the initial
    prevalence for each exposure category.

  * Read any risk-factor-specific settings from ``builder.configuration``.

  * Register value and/or rate modifiers.

  * Register a time-step event handler to, e.g., move people from one exposure
    category to another (if required).

* Some number of value and/or rate modifiers.

* The time-step event handler (if required).

* If you anticipate applying interventions that affect the prevalence of
  exposure categories, it may be convenient to also include a method that
  returns the column name for each exposure category.

Example of an intervention component
------------------------------------

As an example, we will walk through each of these methods for the
:class:`DelayedRisk` risk factor, which was created to model the effects of
tobacco smoking.

The constructor
^^^^^^^^^^^^^^^

This component has one required argument for the constructor, which is the
name of the risk factor.
Note that the constructor defines the default configuration settings for this
component, because this depends on knowing the risk factor's name.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.__init__

The setup method
^^^^^^^^^^^^^^^^

This method performs several necessary house-keeping tasks:

* It reads the configuration settings.

* It loads the initial prevalence for each exposure category.

* It loads the incidence and remission rates, and registers these rates so
  that they will be available at each time-step.

* It request access to the all-cause mortality rate, and loads the relative
  risk of mortality for each exposure category, so that it can determine the
  excess mortality produced by this risk factor.

* It registers rate modifiers for each disease that is affected by this risk
  factor (implemented by the ``register_modifier`` method, described below).

* It loads the disease-specific relative risks for each exposure category.

* It adds an initialization handler to create a column for each exposure
  category, and populate them with the initial prevalence.

* It loads the effects that a tobacco tax would have on the incidence and
  remissions rates.

* It registers an event handler that will be called **before** each time-step
  (by selecting the "time_step__prepare" event) that will move people from one
  post-cessation category to the next.

* It defines the columns that it will need to access, and stores this view in
  ``self.population_view``.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.setup

The initialization method
^^^^^^^^^^^^^^^^^^^^^^^^^

This method creates a column for each of the exposure categories (with
separate columns for the BAU and intervention scenarios), populates them with
the initial prevalence values, and updates the underlying table.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.on_initialize_simulants

The rate modifiers
^^^^^^^^^^^^^^^^^^

This risk factor can affect an arbitrary number of diseases, and so this
component includes the following method, which registers modifiers for:

* The incidence rate of a chronic disease;

* The excess mortality rate of an acute disease/event; and

* The disability rate of an acute disease/event.

This approach was used because the component is currently unable to identify
whether each disease that it affects is a chronic disease or an acute disease.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.register_modifier

The rate modifier method calculates the mean relative risk in the BAU and
intervention scenarios, from which it then calculates the PIF, and modifies
the un-adjusted rate accordingly.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.incidence_adjustment

The prevalence modifier
^^^^^^^^^^^^^^^^^^^^^^^

The prevalence modifier responds to the "time_step__prepare" event, so that
takes effect **before** the time-step itself, and accounts for the normal
transitions between exposure categories in both the BAU and intervention
scenarios:

* The incidence rate moves people from the **never smoked** category to the
  **currently smoking** category;

* The remission rate moves people from the **currently smoking** category to
  the **0 years post-cessation** category; and

* People move from the **N years post-cessation** category to the **N+1 years
  post-cessation** category, until they reach **21+ years post-cessation**.

* It also accounts for mortality in each exposure category.

.. note:: The order in which these transitions are performed is important.
   First, we accumulate people in the final category, **21+ years
   post-cessation**.
   Second, we move people from the **N years post-cessation** category to the
   **N+1 years post-cessation** category in *reverse-chronological order*.
   Finally, we account for incidence and remission.
   This will account for the effects of a tobacco tax in the intervention
   scenario, if the ``tobacco_tax`` configuration setting was set to ``True``,
   and the remission rate in the intervention scenario will be set to zero if
   the ``constant_prevalence``  configuration setting was set to ``True``.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.on_time_step_prepare

The column name method
^^^^^^^^^^^^^^^^^^^^^^

For convenience, this component provides this method that returns a list of
the column names for each exposure category, for both the BAU and intervention
scenarios.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/delay.py
   :pyobject: DelayedRisk.get_bin_names
