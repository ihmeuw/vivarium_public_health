.. _vph_results_concept:

=======
Results
=======

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   observers
   stratification

The ``vivarium_public_health`` results package provides
:term:`components <Component>` for observing and recording public health
measures during a simulation. It builds on vivarium's
:ref:`results management system <results_concept>` — which supplies the
observation and stratification machinery — by adding a standardized base
observer, a set of ready-to-use concrete observers, and a common
stratification component.

The package is organized around two cooperating concerns:

1. **Observers** — a convenience base class
   (:class:`~vivarium_public_health.results.observer.PublicHealthObserver`)
   that wraps the framework's
   :class:`~vivarium.framework.results.observer.Observer` with a simplified
   registration method and a standardized results-formatting pipeline, plus a
   set of ready-to-use concrete observers for common public health measures:
   disability (YLDs), disease state person-time and transition counts,
   mortality (deaths and YLLs), and categorical risk-factor person-time. See
   :ref:`public_health_observer_concept`.
2. **Results stratifier** — a single component that registers the default
   stratifications (age group, sex, current year, event year) shared across
   observers. See :ref:`results_stratifier_concept`.
