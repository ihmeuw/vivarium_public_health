.. _risk_attributable_disease_concept:

=========================
Risk Attributable Disease
=========================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

:class:`~vivarium_public_health.disease.special_disease.RiskAttributableDisease`
provides an alternative to the standard
:ref:`state-machine disease model <disease_model_concept>`. Rather than defining
explicit states and transitions, it derives a simulant's disease status directly
from their exposure to an associated risk factor.

This is a special way to implement a disease model — instead of constructing
states and transitions manually,
``RiskAttributableDisease`` is currently the only implementation for custom
disease models that tie disease status directly to risk factor exposure.

This approach is used for diseases where the condition is **defined by** a risk
threshold — for example, diabetes mellitus defined by fasting plasma glucose
above 7 mmol/L, or protein-energy malnutrition defined by child wasting exposure
categories. In these cases the :term:`population attributable fraction <PAF>` is
effectively 1.

How It Differs from DiseaseModel
---------------------------------

In a standard :class:`~vivarium_public_health.disease.model.DiseaseModel`, you
explicitly create :ref:`states <disease_state_concept>` and
:ref:`transitions <disease_transition_concept>`, then compose them into a model.
A ``RiskAttributableDisease`` handles this automatically:

- **Two states** are auto-created: the disease state (e.g.,
  ``protein_energy_malnutrition``) and a susceptible state (e.g.,
  ``susceptible_to_protein_energy_malnutrition``).
- **Transitions** are determined by whether the simulant's current exposure
  crosses the configured threshold. One forward transition is always created;
  a recovery transition is added if ``recoverable`` is ``True``.
- **No explicit state machine** — the component reads the risk's exposure
  pipeline directly and updates the simulant's disease status on each time step.

Construction requires the full entity strings for cause and risk:

.. code-block:: python

   from vivarium_public_health.disease import RiskAttributableDisease

   rad = RiskAttributableDisease(
       "cause.protein_energy_malnutrition",
       "risk_factor.child_wasting",
   )

Threshold Configuration
-----------------------

The ``threshold`` setting defines which exposure values correspond to the
with-condition state.

Categorical Risks
+++++++++++++++++

For categorical risk factors (dichotomous, ordered polytomous, or unordered
polytomous distributions), provide a **list of category names** whose exposure
indicates disease:

.. code-block:: yaml

   protein_energy_malnutrition:
       threshold: ["cat1", "cat2"]
       mortality: true
       recoverable: true

A simulant whose exposure is in ``cat1`` or ``cat2`` is considered to have
protein-energy malnutrition.

Continuous Risks
++++++++++++++++

For continuous risk factors, provide a **string** with a comparison operator
(``>`` or ``<``) and a numeric value:

.. code-block:: yaml

   diabetes_mellitus:
       threshold: ">7"
       mortality: true
       recoverable: false

A simulant with fasting plasma glucose greater than 7 mmol/L is considered
diabetic. The ``recoverable: false`` setting means that once diagnosed, a
simulant does not lose the condition even if their exposure later drops below
the threshold.

Mortality and Disability
------------------------

``RiskAttributableDisease`` uses the ``ExcessMortalityState`` mixin and
provides the same mortality and disability pipelines as a standard
:class:`~vivarium_public_health.disease.state.DiseaseState`:

- **Disability weight** — loaded from the artifact at
  ``{cause}.disability_weight`` and applied only to simulants with the
  condition.
- **Excess mortality rate** — loaded from the artifact at
  ``cause.{cause_name}.excess_mortality_rate`` and added to the mortality
  rate pipeline.
- **Cause-specific mortality rate** — loaded from the artifact at
  ``cause.{cause_name}.cause_specific_mortality_rate`` and registered as a
  modifier on the ``cause_specific_mortality_rate`` pipeline.

Both EMR and CSMR can be disabled by setting ``mortality: false`` in the
configuration, in which case they default to 0.

Recoverability
--------------

The ``recoverable`` configuration flag controls whether simulants can transition
back to the susceptible state when their exposure falls outside the threshold:

- ``recoverable: true`` — on each time step, simulants whose exposure no longer
  meets the threshold are moved back to the susceptible state.
- ``recoverable: false`` — once a simulant acquires the condition, they remain
  in the with-condition state regardless of future exposure changes. This is
  appropriate for conditions like diabetes that do not resolve simply because
  the defining biomarker improves.

See Also
--------

- :ref:`disease_model_concept`
- :ref:`disease_state_concept`
- :mod:`vivarium_public_health.disease.special_disease`
