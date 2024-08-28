============================================
Non-standard Risk Exposure and Effect Models
============================================

:mod:`vivarium_public_health` provides three components for modeling the impact
of some health attributes on others:

- :class:`~vivarium_public_health.risks.base_risk.Risk`: Model of the
  underlying exposure based on a continuous or categorical distribution.
- :class:`~vivarium_public_health.risks.effect.RiskEffect`: Model of the
  impact of different exposure levels on another health attribute.
- :class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect`: 
  Special-case risk effect model where the risk factors are parameterized
  by exposure levels.

The standard model is to think of exposure to environmental, metabolic, and
behavioral risk factors and their impact on disease incidence rates. However,
we've found many situations to extend this model to other attributes, such as
interventions and their impacts on other risks, diseases, or mortality itself.

In order to support these extended models, we've made the
:class:`~vivarium_public_health.risks.base_risk.Risk`, 
:class:`~vivarium_public_health.risks.effect.RiskEffect`, and
:class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect` components
configurable. This tutorial explains the various configuration options you can
use with these components.

.. contents:
   :local:


Exposure Models
---------------

We model exposure using the
:class:`~vivarium_public_health.risks.base_risk.Risk` or component.
Consider its configuration options:

- ``"exposure"``: This option represents the exposure data source. It defaults
  to the value ``"data"``.
- ``"rebinned_exposure"``: This option tells the component if a categorical
  exposure with more than two categories should be rebinned into
  two categories. It defaults to an empty list, indicating that the
  underlying exposure model should be used.
- ``"category_thresholds"``: This option tells the component how to split
  continuous exposure models into a categorical model. It defaults to an
  empty list, indicating that the underlying exposure model should be used.

The name input when the :class:`~vivarium_public_health.risks.base_risk.Risk`
is created also has an impact on the behavior. Names are provided
as ``<type>.<name>`` where ``type`` refers to the type of entity being
modeled and ``name`` is the name of the entity.  Available types are
``"risk_factor"``, ``"coverage_gap"``, and ``"alternative_risk_factor"``.
Some configuration options are only available for certain entity types, as
summarized in the table below.

.. list-table:: Configuration Options
   :widths: 20 20 20 20
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * -
     - **exposure**
     - **rebinned_exposure**
     - **category_thresholds**
   * - **risk_factor**
     - |check_mark|
     - |check_mark|
     - X
   * - **coverage_gap**
     - |check_mark|
     - X
     - X
   * - **alternative_risk_factor**
     - X
     - X
     - |check_mark|

.. |check_mark| unicode:: U+2713

We'll take each of these entity types one-by-one to see how we can configure
them.


``risk_factor``
+++++++++++++++

For the ``risk_factor`` entity type, both the ``"exposure"`` and
``"rebinned_exposure"`` configuration options are available to us. In the
:ref:`model specification <model_specification_concept>`, we can specify
the component to use its defaults with

.. code-block:: yaml

   components:
       vivarium_public_health:
           risks:
               - Risk("risk_factor.my_risk_factor")

We declare the component but don't declare any configuration options for it.
This will cause the risk component to look up any available exposure
information in the :class:`~vivarium.framework.artifact.artifact.Artifact`
and use the data as presented.

If we change the ``"exposure"`` option to the name of a covariate as

.. code-block:: yaml

   components:
       vivarium_public_health:
           risks:
               - Risk("risk_factor.my_risk_factor")

   configuration:
       my_risk_factor:
           exposure: covariate.my_covariate

the component will look for the covariate estimate in the
:class:`~vivarium.framework.artifact.artifact.Artifact` rather than for
the risk factor exposure. Only covariates with a proportion estimate can be
substituted for risk exposure. The covariate proportion will be used as the
proportion of people exposed to the risk factor.

Finally, we can specify an integer or float value to the ``"exposure"`` option
to directly set the proportion of people exposed.

.. code-block:: yaml

   components:
       vivarium_public_health:
           risks:
               - Risk("risk_factor.my_risk_factor")

   configuration:
       my_risk_factor:
           exposure: 0.6

If the underlying exposure distribution is polytomous (that is, it has
multiple categories of exposure), we can use the ``"rebinned_exposure"`` option
to separate those categories into an "exposed" and "unexposed" category. The
set of categories to rebin into the "exposed" group should be specified as
a list of strings to the ``"rebinned_exposure"`` option.

.. code-block:: yaml

   components:
       vivarium_public_health:
           risks:
               - Risk("risk_factor.my_polytomous_risk_factor")

   configuration:
       my_polytomous_risk_factor:
           rebinned_exposure: ["cat1", "cat2", "cat3"]

This will reformat the exposure data to consider anyone in "cat1", "cat2", or
"cat3" as exposed, and all other exposure categories as unexposed.

Using the ``"rebinned_exposure"`` option will cause the relative risk
for all :class:`~vivarium_public_health.risks.effect.RiskEffect`
components to also be rebinned.

.. note::

   Exposure data is formatted with the typical demographic columns for age,
   sex, location, and year and a value column.  If the exposure data is
   categorical, it also has a "parameter" column with string values of
   "cat1", "cat2", etc.  The categories are presumed to be sorted by severity
   with "cat1" being the worst.


``coverage_gap``
++++++++++++++++

A ``coverage_gap`` entity type is a way of phrasing the lack of coverage of
an intervention as a risk factor.  The only think to keep in mind when
using a coverage gap is what exposure means (1 - intervention coverage).
Otherwise, the configuration options and caveats are the same as
the ``risk_factor`` entity type.

In practice, coverage gaps have a dichotomous distribution, so the
``"rebinned_exposure"`` option does not come into play.


``alternative_risk_factor``
+++++++++++++++++++++++++++

The ``alternative_risk_factor`` is an entity type that indicates we have
both continuous and categorical representations of the exposure. They are used
when an intervention acts on a continuous exposure representation, but the
effects of the exposure are specified in terms of the categorical
exposure representation.

The only relevant configuration option is the ``"category_thresholds"``
option, which **must** be specified. All other keys must be left at their
default values.

.. code-block:: yaml

   components:
       vivarium_public_health:
           risks:
               - Risk("alternative_risk_factor.my_risk_factor")

   configuration:
       my_risk_factor:
           category_thresholds: [7, 8, 9]


The above configuration would correspond to a risk with a continuous exposure.
Individuals in the simulation would be assigned some actual value in this
distribution (e.g. 7.32 or 9.85).  When calculating effects, individuals
would be assigned a category based on which group they sit in, as defined by
the thresholds in the configuration.  The thresholds here correspond to the
groups ``less than 7``, ``between 7 and 8``, ``between 8 and 9``, and
``more than 9``.  For use in determining effect sizes, these groups will be
labelled ``cat1``, ``cat2``, ``cat3``, and ``cat4`` respectively.


Effect Models
-------------

Non-standard effect models can **only** be used with dichotomous exposure
models (models where someone is either exposed or not exposed). The available
configuration options all correspond to generating a relative risk for
the exposed population from a set of parameters.

We model exposure effects using the
:class:`~vivarium_public_health.risks.effect.RiskEffect` or
:class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect` components.

For this tutorial, we'll focus on the ``RiskEffect`` component. The
``NonLogLinearRiskEffect`` component is a special case of the ``RiskEffect``
component where the risk factors are parameterized by exposure levels.

.. todo::
  
   Add details on how to use the ``NonLogLinearRiskEffect`` component.

Let's look at its configuration options:

- ``"relative_risk"``: Option for specifying a relative risk value directly.
  If provided, no other configuration options may be specified.
- ``"mean"``: Option for specifying that the relative risk should be drawn
  from a normal distribution with this mean.  Must also provide a value for
  ``"se"``. No other options may be specified.
- ``"se"``: Option for specifying that the relative risk should be drawn
  from a normal distribution with this standard error.  Must also provide a
  value for ``"mean"``. No other options may be specified.
- ``"log_mean"``: Option for specifying that the relative risk should be drawn
  from a lognormal distribution with this mean.  Must also provide a value for
  ``"log_se"`` and may provide a value for ``"tau_squared"``.  No other
  options may be specified.
- ``"log_se"``: Option for specifying that the relative risk should be drawn
  from a lognormal distribution with this standard error.  Must also provide
  a value for ``"log_mean"`` and may provide a value for ``"tau_squared"``.
  No other options may be specified.
- ``"tau_squared"``: Option for specifying a parameter representing
  inter-study heterogeneity in a lognormal distribution. Can optionally be
  supplied when specifying a relative risk to be drawn with a lognormal
  distribution with ``"log_mean"`` and ``"log_se"``.

When a :class:`~vivarium_public_health.risks.effect.RiskEffect` is created, it
takes two arguments: the name of the exposure model and the name of the
target attribute that should be altered. The exposure model should be named
the same as the argument to :class:`~vivarium_public_health.risks.base_risk.Risk`
and the target attribute should be in the form ``<type>.<name>.<measure>``.
``type`` and ``name`` specify the entity the effect targets and ``measure``
tells the :class:`~vivarium_public_health.risks.effect.RiskEffect` which specific
attribute of the entity to alter. Common targets are exposure for other
:class:`~vivarium_public_health.risks.base_risk.Risk` entities and incidence rates for
diseases.

The Default Case
++++++++++++++++

If we specify no configuration options in the model specification, we end
up with something like:

.. code-block:: yaml

   components:
       vivarium_public_health:
           disease:
               - SIS('my_infectious_disease')
           risks:
               - Risk('risk_factor.my_risk_factor')
               - RiskEffect('risk_factor.my_risk_factor', 'cause.my_infectious_disease.incidence_rate')

In this situation, the :mod:`vivarium_public_health` components will assume
all parameters will come from data.  The
:class:`~vivarium_public_health.disease.models.SIS` component will load measures
like prevalence, incidence rate, excess mortality rate, and others to inform
the initialization and dynamics of the model.  The
:class:`~vivarium_public_health.risks.base_risk.Risk` will load exposure information.
The :class:`~vivarium_public_health.risks.effect.RiskEffect` will load the
population attributable fraction and the relative risk associated with the
risk-cause pair, and link the disease and risk model with this data.

The configuration block for :class:`~vivarium_public_health.risks.effect.RiskEffect`
is specified as

.. code-block:: yaml

   configuration:
       effect_of_<exposure_entity_name>_on_<target_entity_name>:
           <target_entity_measure>:
               ...options...

where ``<exposure_entity_name>`` is the ``<name>`` provided to the associated
:class:`~vivarium_public_health.risks.base_risk.Risk` component and the
``<target_entity_name>`` is the name provided to the component used in
the target, usually another :class:`~vivarium_public_health.risks.base_risk.Risk` or
a disease model.

Specifying a Relative Risk Value
++++++++++++++++++++++++++++++++

If you're in a situation where the size of the effect (the relative risk)
between an exposure model and its target outcome are unknown, one option
is to specify a single value for the relative risk.

.. code-block:: yaml

   components:
       vivarium_public_health:
           disease:
               - SIS('my_infectious_disease')
           risks:
               - Risk('risk_factor.my_risk_factor')
               - RiskEffect('risk_factor.my_risk_factor', 'cause.my_infectious_disease.incidence_rate')

   configuration:
       effect_of_my_risk_factor_on_my_infectious_disease:
           incidence_rate:
               relative_risk: 20

For this to work, the exposure modeled by the
:class:`~vivarium_public_health.risks.base_risk.Risk` must be a dichotomous exposure
(only exposed or not exposed).  The ``"relative_risk"`` option provided will
be assigned and used for the exposed group.  Specifying a relative risk
this way will cause the population attributable fraction to be calculated
using the provided exposure model, and so it does not need to be provided.

Specifying a Relative Risk Distribution
+++++++++++++++++++++++++++++++++++++++

If you have some idea of the uncertainty in the relative risk, you can
specify distribution parameters and have the relative risk value drawn
from that distribution for each simulation.  There are two options for
distributions to use.

The first is to sample from a normal distribution.  You can do so by
providing the following configuration options:

.. code-block:: yaml

   components:
       vivarium_public_health:
           disease:
               - SIS('my_infectious_disease')
           risks:
               - Risk('risk_factor.my_risk_factor')
               - RiskEffect('risk_factor.my_risk_factor', 'cause.my_infectious_disease.incidence_rate')

   configuration:
       effect_of_my_risk_factor_on_my_infectious_disease:
           incidence_rate:
               mean: 10
               se: 3

This will sample a new relative risk from a normal distribution with mean
ten and standard error three in each simulation.  The distribution is clipped
so that values below one are set at one.  Both the ``"mean"`` and ``"se"``
options must be provided.  The ``"mean"`` should be greater than one and the
``"se"`` greater than zero.

A second option is to sample the relative risk from a lognormal distribution.
This can be done with the following configuration options:

.. code-block:: yaml

   components:
       vivarium_public_health:
           disease:
               - SIS('my_infectious_disease')
           risks:
               - Risk('risk_factor.my_risk_factor')
               - RiskEffect('risk_factor.my_risk_factor', 'cause.my_infectious_disease.incidence_rate')

   configuration:
       effect_of_my_risk_factor_on_my_infectious_disease:
           incidence_rate:
               log_mean: 10
               log_se: 3
               tau_squared: 0.5

This will produce a relative risk value:

.. math::

   \textrm{RR} &= \exp(\mu + \sigma X + Y) \\
   X &\sim N(0, 1)\\
   Y &\sim N(0, \tau^2)

The ``"tau_squared"`` parameter is an adjustment for inter-study heterogeneity
and is not required to use the lognormal distribution.

Like the normal distribution, values below one will be clipped and set to one.
All three parameters, the ``"log_mean"``, the ``"log_sd"`` and the
``"tau_squared"``, should be greater than zero if provided.

.. note::

   The parameterized :class:`~vivarium_public_health.risks.effect.RiskEffect` can
   be used with a parameterized version of the
   :class:`vivarium_public_health.risks.base_risk.Risk`.  The only requirement
   for use is that exposure model be dichotomous.
