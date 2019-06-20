Alternative BAU: Constant tobacco prevalence
============================================

We now consider the case where the prevalence of tobacco use in each cohort
remains constant over time --- in other words, the cessation rate is zero.
As per the previous tutorial, the purpose here is to highlight how our
assumptions about the BAU scenario can affect the predicted impact of an
intervention.

The only change that we need to make to the simulation definition is:

+ Set the remission rate to zero.

This is done by adding the following configuration option:

.. code:: yaml

   configuration:
       tobacco:
           constant_prevalence: True

These simulations are already defined in the following files:

+ Tobacco eradication:

  + ``mslt_tobacco_maori_20-years_constant_erad.yaml``
  + ``mslt_tobacco_non-maori_20-years_constant_erad.yaml``

+ Tobacco tax:

  + ``mslt_tobacco_maori_20-years_constant_tax.yaml``
  + ``mslt_tobacco_non-maori_20-years_constant_tax.yaml``

+ Tobacco-free generation:

  + ``mslt_tobacco_maori_20-years_constant_tfg.yaml``
  + ``mslt_tobacco_non-maori_20-years_constant_tfg.yaml``

Intervention comparison
-----------------------

If you run all of these simulations, you can then compare their effects (and
how these differ to those obtained with the original BAU scenario), using the
data analysis software of your choice.

As an example, here are some of the results obtained for non-Maori males aged
50-54 in 2011, for the tobacco eradication intervention:

.. _tobacco_eradication_bau_table3:

.. csv-table:: Results for the tobacco eradication intervention, which yields
   gains in LYs, HALYs, ACMR, and YLDR.
   :file: ../../_static/table_mslt_tobacco_non-maori_20-years_constant_erad_mm.csv
   :header-rows: 1

Note that these results differ to those obtained with the
:ref:`original BAU scenario <tobacco_eradication_bau_table1>`.
