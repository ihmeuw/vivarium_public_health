Writing a custom observer
=========================

As :ref:`explained earlier <concept_observer>`, an observer will typically
record the values in a specific subset of columns at each time-step of a
simulation, and save these data as a single table.
There are three primary concerns when writing a custom observer:

* Deciding which columns to record;

* Recording the data in these columns at each time-step; and

* Collating these data and saving them to an output file.

Structure of an observer component
----------------------------------

.. py:currentmodule:: vivarium_public_health.mslt.observer

An observer component will comprise the following methods:

* A constructor (``__init__``).

* A ``setup(self, builder)`` method that will:

  * Identify which columns to record.

  * Register a time-step event handler to record values at each time-step.

  * Register an end-of-simulation event handler to write the recorded data to
    an output file.

* A time-step event handler.

* An end-of-simulation handler.

Example of an observer component
--------------------------------

As an example, we will walk through each of these methods for the
:class:`MorbidityMortality` observer.
This observer records the core life table quantities (as shown in the
:ref:`example table <example_mslt_table>`) at each year of the simulation.

The constructor
^^^^^^^^^^^^^^^

This component has one required argument for the constructor, which is the
name of the file to which the data will be saved at the end of the simulation:

.. code-block:: yaml

   components:
       vivarium_public_health:
           mslt:
               observer:
                   MorbidityMortality('output_file.csv')

So the ``__init__`` method takes two arguments, and stores the name of the
output file in ``self.output_file``:

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/observer.py
   :pyobject: MorbidityMortality.__init__

The setup method
^^^^^^^^^^^^^^^^

This method performs a several necessary house-keeping tasks:

* It identifies the columns that it will observe.

* It then informs the framework that it will need access to these columns, and
  stores this "view" in ``self.population_view``.

* It stores a reference to the simulation clock in ``self.clock``, so that it
  can determine the current year at each time-step.

* It registers an event handler that will be called **after** each time-step
  (by selecting the "on_collect_metrics" event) that will record the current
  population state.

* It registers an event handler that will be called at the end of the
  simulation (by selecting the "simulation_end" event) that will write the
  recorded data to the output file.

* It creates an empty list, which will contain the data tables recorded at
  each time-step, and stores it in ``self.tables``.

* It defines the column ordering for the output table, and stores it in
  ``self.table_cols``.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/observer.py
   :pyobject: MorbidityMortality.setup

The time-step event handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method records the current values in the specified columns, which is
achieved by:

* Retrieving those columns from the underlying population table, using the
  ``get`` method of ``self.population_view``;

* Checking whether this table contains at least one population cohort;

* Adding a new column, ``year``, to record the current year; and

* Adding this table to the list of recorded tables, ``self.tables``.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/observer.py
   :pyobject: MorbidityMortality.on_collect_metrics

The end-of-simulation event handler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method saves the recorded data, by performing the following steps:

* Concatenating the tables recorded at each time-step into a single table;

* Calculating the year of birth for each cohort, so that individual cohorts
  can be identified by two columns: year of birth, and sex;

* Sorting the table rows so that they are grouped by cohort and arranged
  chronologically;

* Calculating the life expectancy and the health-adjusted life expectancy
  (HALE) for each cohort at each time-step; and

* Writing the sorted table to the specified output file.

.. literalinclude:: ../../../../src/vivarium_public_health/mslt/observer.py
   :pyobject: MorbidityMortality.write_output

.. note:: This is also the appropriate method in which to perform any
   post-processing of the data (e.g., calculating life expectancy and other
   summary statistics).
