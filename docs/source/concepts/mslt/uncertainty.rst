Uncertainty analyses
====================

In order to account for uncertainties in the input data, assumptions about the
business-as-usual scenario, the effects of interventions, etc, we can run many
model simulations --- each using slightly different data --- and see how the
simulation outputs vary as a result of accounting for these sources of
uncertainty.

The basic process is:

1. Identify rate(s) and/or value(s) for which uncertainties exist;

2. Define a probability distribution to characterise the uncertainty for each
   rate/value.

3. Identify whether the samples drawn from each distribution should be
   independent, or correlated in some way. For example, you may wish to
   correlate the samples for each rate across all cohorts (e.g., by age, sex,
   and ethnicity).

4. Draw :math:`N` samples for each of the rate(s) and/or value(s).

5. Store these samples according to the same table structure as per the
   :ref:`original data <mslt_input_data>`, with each sample represented as a
   separate row, and with one additional column (``"draw"``) that identifies
   the sample number (:math:`1 \dots N`).

This will result in a single, larger data artifact that contains all of
samples.
In a model specification, you can then identify both the data artifact **and**
the sample number, and when the simulation is run it will automatically select
the correct values from all data tables that contain multiple samples.

.. note:: Do not create several thousand model specifications that only differ
   in terms of the sample number!
   There are better ways to run multiple simulations.
