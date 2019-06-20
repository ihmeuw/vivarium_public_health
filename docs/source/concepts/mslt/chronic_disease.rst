Chronic disease
===============

To capture intervention effects, we set up multiple parallel diseases as
separate lifetables.
We consider chronic diseases as being independent (i.e., the prevalence of one
disease does not affect the incidence or case fatality rate of another).
The reason for setting up the parallel disease states is that we simulate
intervention effects (through risk factor changes) as changes in disease
incidence rates.
We thus need "BAU" and "intervention" lifetables for all diseases impacted by
the intervention.

The outputs of the chronic disease life tables are:

* A disease-specific mortality rate, for each cohort at each year; and

* A disease-specific YLD rate, for each cohort at each year.

These outputs are generated for both the BAU and intervention scenarios, with
the difference between BAU and intervention (across all of the disease life
tables) then being subtracted from the BAU all-cause mortality and morbidity
rates, to create the "intervention" life table.
We can then measure the intervention effect in terms of the differences in
LYs, HALYs, LE, and HALE, between the BAU and intervention life tables.

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

   "If it is assumed that mortality from all other causes is independent of
   the disease, i.e., that it is the same for healthy and diseased people,
   this implies that the transition hazards for incidence, remission and case
   fatality are not affected by the value of the 'all other causes' mortality.
   Therefore we can set the value of mortality from all other causes to 0
   (i.e., leave it out of the equations) and still derive the right values for
   the disease rates."

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
