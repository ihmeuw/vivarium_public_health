Constant tobacco prevalence
===========================

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

.. todo:: Show results as per Table 4 of the manuscript.
