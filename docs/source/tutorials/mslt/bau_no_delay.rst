Immediate recovery upon cessation
=================================

We now consider the case where cessation of smoking results in **immediate**
recovery, rather than taking 20 years for the tobacco-associated relative
risks to decrease back to 1.0.
The purpose here is to highlight how our assumptions about the BAU scenario
can affect the predicted impact of an intervention.

The only changes that we need to make to the simulation definition are:

1. To use a different data artifact for these simulations, where the initial
   prevalence of tobacco use is only defined for 3 exposure levels: never
   smoked, current smoker, and former smoker; and

2. Set the recovery delay to 0 years.

.. note:: We could have used the same data artifact as in previous
   simulations, but then the tobacco component would have to manipulate the
   input data into the appropriate form. We instead choose to perform all
   input data manipulation *before* generating the data artifacts.

.. code:: yaml

   configuration:
       input_data:
           # Change this to "mslt_tobacco_maori_data_0-years.hdf" for the Maori
           # population.
           artifact_path: mslt_tobacco_non-maori_0-years.hdf
       # Other configuration settings ...
       tobacco:
           delay: 0

These simulations are already defined in the following files:

+ Tobacco eradication:

  + ``mslt_tobacco_maori_0-years_decreasing_erad.yaml``
  + ``mslt_tobacco_non-maori_0-years_decreasing_erad.yaml``

+ Tobacco tax:

  + ``mslt_tobacco_maori_0-years_decreasing_tax.yaml``
  + ``mslt_tobacco_non-maori_0-years_decreasing_tax.yaml``

+ Tobacco-free generation:

  + ``mslt_tobacco_maori_0-years_decreasing_tfg.yaml``
  + ``mslt_tobacco_non-maori_0-years_decreasing_tfg.yaml``

Intervention comparison
-----------------------

.. todo:: Show results as per Table 4 of the manuscript.
