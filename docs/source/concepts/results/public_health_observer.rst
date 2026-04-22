.. _public_health_observer_concept:

======================
Public Health Observer
======================

.. contents::
   :depth: 3
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.results.observer.PublicHealthObserver` is a
convenience base class for building :ref:`observers <results_concept>` in
public health simulations. It extends vivarium's
:class:`~vivarium.framework.results.observer.Observer` with two capabilities
that most public health observers share:

1. A simplified method for registering
   :class:`adding observations <vivarium.framework.results.observation.AddingObservation>`.
2. A standardized results-formatting pipeline that produces a consistent
   column layout across all public health outputs.

All of the :ref:`concrete observers <concrete_observers_concept>` shipped with
this package inherit from ``PublicHealthObserver``.

Registering Adding Observations
-------------------------------

The most common observation type in public health models is the *adding*
observation: one that sums new results into a running total each time step
(e.g. counting deaths or accumulating person-time). Rather than calling the
:ref:`results interface <results_concept>` directly (via the builder),
``PublicHealthObserver`` exposes
:meth:`~vivarium_public_health.results.observer.PublicHealthObserver.register_adding_observation`,
which wraps that call with sensible defaults and automatically applies the
standardized formatter.

The method accepts the same core arguments as
:meth:`~vivarium.framework.results.interface.ResultsInterface.register_adding_observation`
(``pop_filter``, ``when``, ``aggregator``, etc.) plus two convenience
parameters:

- ``additional_stratifications`` — extra stratification names to add on top
  of the defaults registered by the
  :ref:`ResultsStratifier <results_stratifier_concept>`.
- ``excluded_stratifications`` — default stratification names to remove from
  this particular observation.

Results Formatting
------------------

Public health result files follow a standardized column layout. Every row
contains four metadata columns in addition to any stratification and value
columns:

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Column
     - Purpose
   * - ``measure``
     - The name of the quantity being measured (e.g. ``"person_time"``,
       ``"transition_count"``).
   * - ``entity_type``
     - A classifier for the entity (e.g. ``"cause"``, ``"rei"``).
   * - ``entity``
     - The specific entity being observed (e.g. ``"measles"``,
       ``"high_systolic_blood_pressure"``).
   * - ``sub_entity``
     - A finer-grained descriptor within the entity (e.g. a specific
       disease state or risk category).

``PublicHealthObserver`` produces these columns through a chain of overridable
methods that subclasses customize:

- :meth:`~vivarium_public_health.results.observer.PublicHealthObserver.format` —
  general-purpose reshaping of the raw results (default: ``reset_index``).
- :meth:`~vivarium_public_health.results.observer.PublicHealthObserver.get_measure_column` —
  returns the ``measure`` values (default: the observation name).
- :meth:`~vivarium_public_health.results.observer.PublicHealthObserver.get_entity_type_column` —
  returns the ``entity_type`` values (default: empty string).
- :meth:`~vivarium_public_health.results.observer.PublicHealthObserver.get_entity_column` —
  returns the ``entity`` values (default: empty string).
- :meth:`~vivarium_public_health.results.observer.PublicHealthObserver.get_sub_entity_column` —
  returns the ``sub_entity`` values (default: empty string).

The top-level
:meth:`~vivarium_public_health.results.observer.PublicHealthObserver.format_results`
method calls each of these in sequence and reorders the final columns so that
metadata columns appear first, stratification columns in the middle, and the
``value`` column last.

Writing a Custom Observer
-------------------------

To create a new public health observer:

1. Subclass ``PublicHealthObserver``.
2. Implement
   :meth:`~vivarium.framework.results.observer.Observer.register_observations`
   and call ``self.register_adding_observation(...)`` within it.
3. Override the formatting sub-methods as needed to populate the metadata
   columns.

.. _concrete_observers_concept:

Concrete Observers
------------------

The ``vivarium_public_health`` results package ships several concrete
observers that cover the most common public health measures. Each inherits
from :class:`~vivarium_public_health.results.observer.PublicHealthObserver`
and registers one or more
:class:`adding observations <vivarium.framework.results.observation.AddingObservation>`
during setup.

All concrete observers support per-observer stratification overrides via the
``stratification`` configuration block. Under
``configuration.stratification.<observer_config_name>``, each observer
accepts an ``include`` list of additional stratification names and an
``exclude`` list of default stratification names to remove.

.. _disability_observer_concept:

Disability Observer
+++++++++++++++++++

:class:`~vivarium_public_health.results.disability.DisabilityObserver` counts
:term:`years lived with disability <Years Lived with Disability>` (YLDs).

It discovers all components that contribute disability weights — by default
:class:`~vivarium_public_health.disease.state.DiseaseState` and
:class:`~vivarium_public_health.disease.special_disease.RiskAttributableDisease`
instances — and adds an ``all_causes`` aggregate on top.

**Observation registered:** ``ylds``

On each ``time_step__prepare`` event, the observer reads each cause's
``{cause}.disability_weight`` pipeline for every living simulant, multiplies
by the step size (converted to years), and sums the result. Because a simulant
can carry disability from multiple causes simultaneously, YLDs are *not*
stratified by cause through the normal stratification mechanism. Instead, the
observer produces wide results (one column per cause) and then reshapes them
into long format during formatting, with each cause appearing as a separate
``sub_entity``.

Categories listed under ``excluded_categories.disability`` in the model
specification are dropped before observation.

The observer's configuration key is ``disability``.

.. _disease_observer_concept:

Disease Observer
++++++++++++++++

:class:`~vivarium_public_health.results.disease.DiseaseObserver` tracks
disease-state person time and transition counts for a single disease model.

Each instance is constructed with the name of the disease to observe
(e.g. ``DiseaseObserver("measles")``).

**Observations registered:**

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Observation
     - Description
   * - ``person_time_{disease}``
     - Person-time (in years) spent in each disease state during each time
       step, observed at ``time_step__prepare``.
   * - ``transition_count_{disease}``
     - Count of simulants that transitioned between disease states during
       each time step, observed at ``collect_metrics``.

In addition to registering observations, the observer registers two
stratifications specific to its disease:

1. **Disease state** — categories correspond to the state IDs of the
   disease model's states (e.g. ``susceptible_to_measles``, ``measles``).
2. **Transition** — categories correspond to the transition names
   (e.g. ``susceptible_to_measles_to_measles``), plus a ``no_transition``
   category that is automatically excluded from results.

To track transitions, the observer maintains a ``previous_{disease}`` column
on the :ref:`state table <population_concept>` that is updated at the start
of each time step.

The observer's configuration key matches the disease name (e.g.
``configuration.stratification.measles``).

.. _mortality_observer_concept:

Mortality Observer
++++++++++++++++++

:class:`~vivarium_public_health.results.mortality.MortalityObserver` counts
cause-specific deaths and :term:`years of life lost <Years of Life Lost>`
(YLLs).

It discovers all components that use the ``ExcessMortalityState`` mixin and
have active excess mortality, then adds ``other_causes`` (for non-modeled
causes) and ``not_dead`` (automatically excluded) as additional categories.

**Observations registered:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Observation
     - Description
   * - ``deaths``
     - Count of simulants who died during the current time step
       (``exit_time > clock()``).
   * - ``ylls``
     - Sum of ``years_of_life_lost`` for simulants who died during the
       current time step.

Both observations filter on ``is_alive == False`` and are observed at
``collect_metrics``.

By default the observer registers a ``cause_of_death`` stratification so that
deaths and YLLs are broken down by cause. Setting ``aggregate: true`` in the
configuration collapses all causes into a single ``all_causes`` row, which can
improve runtime when cause-level detail is not needed. Set
``configuration.stratification.mortality.aggregate`` to ``true`` to enable
this mode.

The observer's configuration key is ``mortality``.

.. _categorical_risk_observer_concept:

Categorical Risk Observer
+++++++++++++++++++++++++

:class:`~vivarium_public_health.results.causal_factor.CategoricalRiskObserver`
tracks person-time spent in each exposure category of a categorical risk
factor. It is a convenience subclass of
:class:`~vivarium_public_health.results.causal_factor.CategoricalCausalFactorObserver`,
which provides the core logic.

Each instance is constructed with the name of the risk factor
(e.g. ``CategoricalRiskObserver("child_wasting")``).

**Observation registered:** ``person_time_{risk_factor}``

At ``time_step__prepare`` the observer counts the number of living simulants
in each exposure category and multiplies by the step size (converted to
years) to produce person-time.

The observer registers a stratification for the risk factor whose categories
are loaded from the artifact at ``risk_factor.{name}.categories``, and the
exposure values come from the ``{name}.exposure`` pipeline.

The observer's configuration key matches the risk factor name (e.g.
``configuration.stratification.child_wasting``).

See Also
--------

- :ref:`results_concept` — vivarium's results management system
- :ref:`results_stratifier_concept` — common public health stratifications
- :mod:`vivarium_public_health.results.observer`
- :mod:`vivarium_public_health.results.disability`
- :mod:`vivarium_public_health.results.disease`
- :mod:`vivarium_public_health.results.mortality`
- :mod:`vivarium_public_health.results.causal_factor`
