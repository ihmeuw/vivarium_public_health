==============================================
Using Data Free Risk + RiskEffect Components
==============================================

With version 0.8.11 of vivarium_public_health, we now support the ability to specify values for risk exposure and relative risk directly from the model configuration rather than requiring that data to be in an artifact. 

RISK EXPOSURE
--------------------------

To specify risk exposure via the configuration file, you have two options. You can either provide a single value between 0 and 1, inclusive, or provide the name of a covariate to use as a proxy. Note that supplying data via the config is currently only supported for dichotomous risks. The data that you supply in the configuration will be used to calculate the exposure values for cat1, with cat2 calculated as 1-cat1. Thus if you provide 0.3 as the exposure value in the config, the exposure for cat1 will be 0.3 and that for cat2 will be 0.7. If you use a covariate as a proxy, you can specify either the name of the covariate (e.g., antenatal_care_1_visit_coverage_proportion) to use the covariate estimates as exposure values for cat1 or 1-covariate (e.g., 1 - antenatal_care_1_visit_coverage_proportion) to use the one minus the covariate estimates as the exposure values (and thus the covariate estimates as the exposure values for cat2). 

For a coverage gap named 'my_coverage_gap' where you wanted to use antenatal_care_1_visit_coverage_proportion as a proxy for the exposure values, your model specification would contain the following blocks:

.. code-block:: yaml

    components:
        vivarium_public_health:
                risks:
                        - Risk("coverage_gap.my_coverage_gap")

    configuration:
        my_coverage_gap:
                exposure: antenatal_care_1_visit_coverage_proportion



RISK EFFECTS
--------------------------

To use the configuration to supply data about the effect of a risk on some other entity+measure, you can specify the relative risk for cat1 as a value similarly to how we specified the exposure value above or you can supply parameters for a distribution from which RR will be sampled. In the configuration section of your model specification, you should include a block labeled effect_of_<RISK_NAME>_on_<AFFECTED_ENTITY> (e.g., effect_of_unsafe_water_source_on_diarrheal_diseases). Within that block, you will add another sub-block for each measure you wish to affect.  For a risk/coverage gap acting on another risk/coverage gap, this sub-block should be called exposure_parameters. For a risk/coverage_gap acting on a cause, it should be incidence_rate or excess_mortality depending on what you want to affect. Within the measure block, you can then either specify a single value or the set of parameters for the distribution as described below.  If you do not specify either a value or parameters for a distribution, the simulation will use data in the artifact by default.


Specifying a single value
--------------------------

If you specify a value, note that only a single value (between 1 and 100, inclusive) is allowed - there is no option to use a proxy as with covariates for exposure. To specify a value, you will add a 'relative_risk' block with your value. 

For example, to add a RiskEffect of my_coverage_gap on the incidence rate of my_cause, you would have the following in your model specification: 

.. code-block:: yaml

    components:
        vivarium_public_health:
                risks:
                        - Risk("coverage_gap.my_coverage_gap")
                        - RiskEffect("coverage_gap.my_coverage_gap", "cause.my_cause.incidence_rate")

    configuration:
        my_coverage_gap:
                exposure: antenatal_care_1_visit_coverage_proportion
        effect_of_my_coverage_gap_on_my_cause:
                incidence_rate:
                        relative_risk: 50



Specifying distribution parameters
-----------------------------------
There are two sets of distribution parameters you can specify to create a distribution from which a draw will be sampled. Currently, this draw will not vary by demographic groups. This functionality was added based on what was being done in Auxiliary Data Processing to create relative risks and all currently existing coverage gaps were using one of these two sets of distribution parameters and a single value for each draw for all demographic groups.


    1.) You can specify mean and standard error and relative risk will be drawn from a normal distribution with that mean and standard error. For example:

    .. code-block:: yaml

        components:
            vivarium_public_health:
                    risks:
                            - Risk("coverage_gap.my_coverage_gap")
                            - RiskEffect("coverage_gap.my_coverage_gap", "cause.my_cause.incidence_rate")

        configuration:
            my_coverage_gap:
                    exposure: antenatal_care_1_visit_coverage_proportion
            effect_of_my_coverage_gap_on_my_cause:
                    incidence_rate:
                            mean: 5
                            se: 0.5


    This will use numpy.random.normal(mean, se) to draw a value for the relative risk. 

    2.) You can specify mean and standard error of the log distribution as well as tau_squared, the interstudy heterogeneity. For example: 

    .. code-block:: yaml

        components:
            vivarium_public_health:
                    risks:
                            - Risk("coverage_gap.my_coverage_gap")
                            - RiskEffect("coverage_gap.my_coverage_gap", "cause.my_cause.incidence_rate")

        configuration:
            my_coverage_gap:
                    exposure: antenatal_care_1_visit_coverage_proportion
            effect_of_my_coverage_gap_on_my_cause:
                    incidence_rate:
                            log_mean: 5
                            log_se: 0.5
                            tau_squared: 0.1


    As was done in ADP, this will draw a value as numpy.exp(log_se * numpy.random.rand() + log_mean + numpy.random.normal(0, tau_squared))

Either distribution format will floor the values at 1 as was done in ADP. 



