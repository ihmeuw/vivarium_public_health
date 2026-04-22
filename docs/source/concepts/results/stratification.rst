.. _results_stratifier_concept:

==================
Results Stratifier
==================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.results.stratification.ResultsStratifier`
is a :term:`component <Component>` that registers the common
:ref:`stratifications <results_concept>` shared across public health observers.
It serves as the single place where default grouping dimensions — age, sex,
and time — are defined for the simulation's results.

Concrete observers such as
:ref:`DiseaseObserver and MortalityObserver <concrete_observers_concept>` may
register additional observer-specific stratifications, but the broadly
applicable ones live here.

Default Stratifications
-----------------------

``ResultsStratifier`` registers the following stratifications during setup:

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Name
     - Source Column(s)
     - Description
   * - ``age_group``
     - ``age``
     - Bins simulant ages into age-group name strings (e.g.
       ``"early_neonatal"``, ``"1_to_4"``). Bins are derived from
       ``population.age_bins`` in the artifact and filtered to the
       configured ``initialization_age_min`` and ``untracking_age``
       range.
   * - ``current_year``
     - ``current_time``
     - Maps each time step's clock time to a calendar-year string (e.g.
       ``"2025"``). Categories span from the configured start year to
       the end year (inclusive).
   * - ``event_year``
     - ``event_time``
     - Maps the event time to a calendar-year string. The event time is
       the end of the current time step (i.e.
       ``current_time + step_size``), which may fall in the next
       calendar year relative to ``current_time``.
       Categories span start year to end year + 1,
       with the extra year automatically excluded so it does not appear
       in output.
   * - ``sex``
     - ``sex``
     - Groups simulants by ``"Female"`` or ``"Male"``. No mapper is
       needed because the column values already match the category
       names.

Activating Stratifications
---------------------------

Registering a stratification makes it *available* but does not apply it to
any observation automatically. To activate stratifications for all observations, list them under
the ``stratification.default`` key in the
:ref:`model specification <model_specification_concept>` configuration.
The key path is ``configuration.stratification.default``, whose value is a
list of stratification names (e.g. ``"age_group"``, ``"sex"``,
``"current_year"``).

Individual observers can further customize which defaults they use through
their ``include`` and ``exclude`` configuration keys (see
:ref:`concrete_observers_concept`).

Age Bins
--------

Age-group bins are loaded from the ``population.age_bins`` artifact key and
trimmed to the simulation's configured age window
(``population.initialization_age_min`` through ``population.untracking_age``).
Group names are normalized to lowercase with underscores (e.g.
``"Early Neonatal"`` becomes ``"early_neonatal"``).

The mapper uses :func:`pandas.cut` to assign each simulant's continuous age to
the appropriate bin.

Mappers
-------

Two mappers are provided:

- :meth:`~vivarium_public_health.results.stratification.ResultsStratifier.map_age_groups` —
  vectorized mapper that bins ages into age-group name strings.
- :meth:`~vivarium_public_health.results.stratification.ResultsStratifier.map_year` —
  vectorized mapper that extracts the year from a datetime column and returns
  it as a string.

Both mappers are vectorized (``is_vectorized=True``), meaning they operate on
the entire population DataFrame at once rather than row by row.

See Also
--------

- :ref:`results_concept` — vivarium's results management system
- :ref:`public_health_observer_concept` — the base observer class
- :ref:`concrete_observers_concept` — the concrete public health observers
- :mod:`vivarium_public_health.results.stratification`
