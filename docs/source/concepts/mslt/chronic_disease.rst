Chronic disease
===============

To account for disease-specific morbidity and mortality, we consider chronic
diseases as being **independent** (i.e., prevalence of one disease does not
affect that of another) and which modify the all-cause mortality and YLD
rates.

The outputs of a chronic disease are:

* A disease-specific mortality rate; and

* A disease-specific YLD rate.

These disease-specific rates in the "business-as-usual" (BAU) scenario are
subtracted from the all-cause rates, to produce cause-deleted mortality and YLD
rates (i.e., the mortality and YLD rates due to all causes **except** those
explicitly modelled as arising from a specific cause).
Each chronic disease then contributes these disease-specific rates, which are
added to the cause-deleted mortality and YLD rates to obtain the all-cause
rates.

This has the following implications:

* In the BAU scenario, modelling one or more diseases explicitly **has no
  effect** on the all-cause mortality and YLD rates, since the
  disease-specific rates are subtracted from the all-cause rates, and are then
  added to the cause-deleted rates.

* An intervention that affects the prevalence or impact of a disease (e.g., by
  reducing its incidence rate) **will have an effect** on the all-cause
  mortality and YLD rates. This is because the disease-specific rates **for
  the BAU** are subtracted from the all-cause rates, but the disease-specific
  rates **for the intervention scenario** are added to the cause-deleted
  rates.

A chronic disease is characterised in terms of:

* Incidence rate (:math:`i`);
* Remission rate (:math:`r`);
* Case fatality rate (:math:`f`);
* Initial prevalence (:math:`C(0)`); and
* Disability rate.

The equations for chronic disease prevalence, remission, and mortality come
from `Barendregt et al., 2003 <https://doi.org/10.1186/1478-7954-1-4>`_.
A key assumption in their derivation is the independence of mortality from all
causes:

   If it is assumed that mortality from all other causes is independent of the
   disease, i.e., that it is the same for healthy and diseased people, this
   implies that the transition hazards for incidence, remission and case
   fatality are not affected by the value of the 'all other causes' mortality.
   Therefore we can set the value of mortality from all other causes to 0
   (i.e., leave it out of the equations) and still derive the right values for
   the disease rates.

With this simplifying assumption, the system of equations are:

.. math::

   \begin{align}
     \DeclareMathOperator{\d}{d\!}
     \frac{\d{}S_a}{\d{}a} &= -i_a S_a + r_a C_a \\
     \frac{\d{}C_a}{\d{}a} &= -(f_a + r_a) C_a + i_a S_a \\
     \frac{\d{}D_a}{\d{}a} &= f_a C_a
   \end{align}

.. table:: Definition of symbols used in the chronic disease equations.

   ===========  ============================================================
   Symbol       Definition
   ===========  ============================================================
   :math:`i_a`  Disease incidence rate for people of age :math:`a`.
   :math:`r_r`  Disease remission rate for people of age :math:`a`.
   :math:`f_a`  Case fatality rate for people of age :math:`a`.
   :math:`S_a`  Number of healthy people at age :math:`a`.
   :math:`C_a`  Number of diseased people at age :math:`a`.
   :math:`D_a`  Number of dead people at age :math:`a` (due to the disease).
   ===========  ============================================================

This is a system of linear ordinary differential equations (ODEs), for which
an analytic solution can be obtained (see equations (4)--(6) in
`Barendregt et al., 2003 <https://doi.org/10.1186/1478-7954-1-4>`_).
