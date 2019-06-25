.. _concept_intervention:

Interventions
=============

We will consider three different interventions that affect the prevalence of
tobacco smoking.

Each of these interventions will affect the exposure distribution of the risk
factor (tobacco smoking).
This will be done by modifying any of the rates that affect the exposure
(i.e.,, the uptake and remission rates), or by moving people from one exposure
category to another.

.. note:: Another option, not explored here, is to modify the relative risk(s)
   associated with an exposure category (the "relative risk shift" method,
   `Barendregt and Veerman, 2009 <https://doi.org/10.1136/jech.2009.090274>`_).
   With this method, proportions of the cohort do not transition between
   exposure states.
   Rather, each exposure category has a shift in its average exposure which is
   modelled as a shift in its relative risk.
   We use this method for BMI categories, but for smoking we explicitly model
   transitions between smoking states.

Tobacco eradication
-------------------

For this intervention, we assume that tobacco is no longer available from some
specific year :math:`Y`.
This will have two effects:

* From year :math:`Y`, the uptake rate will be zero; and

* At year :math:`Y`, all current smokers will cease to smoke and their
  exposure category will be changed to **0 years post-cessation**.
  They will then progress through the post-cessation exposure categories and,
  20 years later, they will have the same disease incidence rates as the
  **never smoked** exposure category.

Tobacco-free generation
-----------------------

For this intervention, we assume that individuals born after a certain year
:math:`Y` will be unable to purchase tobacco and therefore will never smoke.
This will have one effect (where we assume that all uptake occurs at age 20):

* From year :math:`Y + 20`, the uptake rate will be zero.

Tobacco tax
-----------

This is a more complex intervention, where we assume that there will be a
gradual tax increase that affects the price of cigarette packs, and that
tobacco uptake and cessation will be affected by the annual cost increase.

While the underlying details are more complex than the other interventions
outlined above, the effects of this intervention on tobacco smoking prevalence
are themselves simple:

* The uptake rate will be reduced by some proportion; and

* The cessation rate will be increased by some proportion.

The reduction in uptake will grow larger over time, since the tobacco price
will increase over time.
However, the impact on cessation rates is only felt in the year of tax
increase
(`Blakely et al., 2015 <https://doi.org/10.1371/journal.pmed.1001856>`_).

.. note:: The size of these effects is determined by price elasticities, which
   can vary by sex and age (and other strata of heterogeneity, as required).
