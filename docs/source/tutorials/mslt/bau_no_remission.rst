Constant tobacco prevalence
===========================

.. todo:: Describe this scenario and show how it affects the tobacco
   interventions.
   The purpose is to highlight how our assumptions about the BAU scenario can
   affect the predicted impact of an intervention.

.. note:: This **does not** require a different data artifact, because it only
   sets the remission rate to zero.
   This adjustment **could not** be achieved with an intervention component,
   because it would only take effect in the intervention scenario, and we need
   this adjustment to apply to both the intervention **and** BAU scenarios.

.. code:: yaml

   configuration:
       tobacco:
           constant_prevalence: True
