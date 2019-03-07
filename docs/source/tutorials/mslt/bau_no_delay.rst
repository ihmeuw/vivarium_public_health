Immediate recovery upon cessation
=================================

.. todo:: Describe this scenario and show how it affects the tobacco
   interventions.
   The purpose is to highlight how our assumptions about the BAU scenario can
   affect the predicted impact of an intervention.

.. note:: This requires a different data artifact, because the tobacco
   component has only 3 exposure levels (never smoked, current smoker, and
   former smoker) and so the initial prevalence of tobacco use in the
   population is necessarily different from the delayed-recovery scenario.

.. code:: yaml

   configuration:
       input_data:
           # Change this to "mslt_maori_data_no-delay.hdf" for the Maori population.
           artifact_path: mslt_non-maori_data_no-delay.hdf
       # Other configuration settings ...
       tobacco:
           delay: 0
