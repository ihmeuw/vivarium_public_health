.. _mslt_tutorials:

Multi-State Life Tables: chronic disease interventions
======================================================

In this tutorial, you will learn how to reproduce each of the simulations
presented in the paper "Multistate lifetable modelling of preventive
interventions: Concept and Python code".

After completing this tutorial, you will be able to adapt and modify these
simulations, to explore the impact of different model assumptions and
interventions, and to capture different simulation outputs of interest.

.. todo:: Provide the necessary data artifact(s) and simulation files in an
   accompanying repository.

   They are currently hosted here: https://github.com/collijk/mslt_port/

.. toctree::
   :maxdepth: 2
   :caption: First steps

   bau
   chd

.. toctree::
   :maxdepth: 2
   :caption: Tobacco interventions

   tobacco_bau
   tobacco_results

.. toctree::
   :maxdepth: 2
   :caption: Different BAU scenarios

   bau_no_delay
   bau_no_remission

.. toctree::
   :maxdepth: 2
   :caption: Writing your own MSLT components

   custom_intervention
   custom_observer
   custom_risk_factor
