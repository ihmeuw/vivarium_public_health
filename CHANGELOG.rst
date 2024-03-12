**2.3.1 - 3/11/24**

 - Update Mortality Observer to include tracked in population filter
 - Fix bug in get_initialization_parameters to only remove existing keys if necessary

**2.3.0 - 3/7/24**

 - Update population configuration keys to be more descriptive

**2.2.3 - 3/6/24**

 - Update Mortality Observer to allow running with cause specific or total deaths and ylls.

**2.2.2 - 2/28/24**

 - Fix bug in rescale_binned_proportions to update midpoitn for new age bins

**2.2.1 - 2/26/24**

 - Update LinearScaleUp configuration defaults

**2.2.0 - 02/14/24**

 - Implement CausesConfigurationParser to parse causes configuration into DiseaseModel components
 - Bugfix assign sex-location-age demographic proportions by year when only one year in dataset
 
**2.1.4 - 01/10/24**

 - Exclude undesirable arguments from the return of `BaseDiseaseState` `name` and `__repr__` methods
 
**2.1.3 - 01/09/24**

 - Update PyPI to 2FA with trusted publisher

**2.1.2 - 12/21/23**

 - Fix tests failing due to Vivarium 2.3.0 release

**2.1.1 - 10/13/23**

 - Perform actions in DiseaseState setup using class methods rather than hardcoding to allow for cleaner subclassing

**2.1.0 - 10/05/23**

 - Remove explicit support for Python 3.8
 - Minor bugfix to ensure default remission rate calls the right artifact key

**2.0.1 - 09/27/23**

 - Address a CopyWithSettingWarning in results stratifier

**2.0.0 - 09/22/23**

 - Refactor all components to inherit from `vivarium.Component`
 - Refactor components inheriting from another vivarium class to be compatible with vivarium 2.0.0

**1.0.5 - 09/19/23**

 - Update unit test for dtypes

**1.0.4 - 09/15/23**

 - Address Pandas 2.1 FutureWarnings

**1.0.3 - 08/10/23**

 - Pass `BaseDiseaseState` constructor kwargs to its super-class's constructor

**1.0.2 - 08/10/23**

 - Minor bugfix to ensure dead simulants do not get observed transitions

**1.0.1 - 08/07/23**

 - Minor bugfix to improve handling of excess mortality rate data

**1.0.0 - 08/02/23**

 - Performance and architectural improvements to results manager, including observers
 - Updates versioning to use setuptools_scm
 - Other bugfixes

**0.11.0 - 06/01/23**

 - Support Python 3.8-3.11
 - Update vivarium pin
 - Handle FutureWarning
 - Refactor to create a 'get_transition_names' function

**0.10.24 - 05/11/23**

 - Standardize builder, cause argument order in state get data functions
 - Mends a bug where configured key_columns for randomness were not used in register_simulants

**0.10.23 - 05/03/23**

 - Throw error when artifact doesn't contain relative risk data for desired target
 - Rename `for_initialization` argument to match updated argument name in vivarium

**0.10.22 - 12/27/22**

 - Update CI and setup for building python 3.7-3.10

**0.10.21 - 11/16/22**

 - Fix bug in timing of disease transition observations
 - Add logging when adding risks with a relative risk less than 1 from artifact

**0.10.20 - 07/25/22**

 - Update Vivarium pin

**0.10.19 - 06/29/22**

 - Create new LBWSG components
 - Fix a bug when stratifying newly born simulants
 - Fix pandas deprecation warnings
 - Fix a bug when stratifying an empty population
 - Allow configuration of sex subsetting of the population
 - Add support for empty populations
 - Fix a bug in counting deaths and ylls
 - Refactor DiseaseState to be compatible with latest release of vivarium
 - Add CODEOWNERS

**0.10.18 - 04/22/22**

 - Improve ScaleUp component configuration
 - Enable Mortality component to handle affected unmodeled causes
 - Refactor RiskEffect calculation for clarity and extensibility
 - Implement ResultsStratifier to stratify outputs
 - Refactor all observers to be compatible with the ResultsStratifier component

**0.10.17 - 02/15/22**

 - Autoformat code with black and isort.
 - Add black and isort checks to CI.

**0.10.16 - 02/13/22**

 - Update CI

**0.10.15 - 01/25/22**
 - Implement LinearScaleUp component
 - Refactor Risk, RiskEffect, and Mortality components for inheritance
 - Added pull request template
 - Fix bug in excess mortality pipeline name
 - Fix bug in risk propensity pipeline

**0.10.14 - 10/29/21**
 - Update license to BSD 3-clause
 - Add .zenodo.json metadata replacing AUTHORS.rst

**0.10.13 - 08/31/21**
 - implement categorical risk observer
 - fix pandas warning in application of risk effect
 - fix column name bug
 - improve performance of reshaping
 - require 2.0.6 and later of risk_distributions

**0.10.12 - 08/10/21**
 - Fix bugs in DiseaseState
 - Improve functionality of ensemble distributions
 - Improve CI

**0.10.11 - 05/18/21**
 - Fix bug in computing ages from an age distribution

**0.10.10 - 05/10/21**
 - Improve standard DiseaseObserver
 - Add 'transition rate' to the RateTransition object
 - Add state and transition names to DiseaseModel and RiskAttributableDisease
 - Get location from artifact rather than config file
 - Fix bug that resulted in non-unique initializations of populations

**0.10.9 - 01/25/21**
 - Improve performance of polytomous risk ppf calculations

**0.10.8 - 1/5/21**
 - Fix deploy script

**0.10.7 - 1/5/21**
 - Github actions replaces Travis for CI
 - Unpin pandas and numpy

**0.10.6 - 11/5/20**
 - Fix bug when risk effects are defined by a distribution

**0.10.5 - 10/2/20**
 - Remove code from shigella vaccine
 - Remove sample history observer
 - Update randomness implementation to be consistent with latest version of
   vivarium
 - Make prevalence sampling configurable
 - Refactor to avoid warnings
 - Clarify cut age bin math
 - Pin to pandas 0.24.x
 - Fix Travis validation issues

**0.10.4 - 01/14/20**

 - Fix regression bug in RiskAttributableDisease
 - Introduce low birth weight and short gestation risk and risk effect

**0.10.3 - 12/13/19**

 - Fix regression bug in SIR_fixed_duration.
 
**0.10.2 - 11/29/19**

 - Fix disease observer bug that prevented it from loading its configuration.

**0.10.1 - 11/27/19**

 - Update MSLT components to new vivarium APIs.

**0.10.0 - 11/18/19**

 - Update vivarium event system usage to no longer require explicit use of
   events.
 - Move Artifact to vivarium.
 - Clean up utility functions location and usage.
 - Consistent preference of pathlib over os.path
 - Small API updates for configuration.
 - Restructure components to allow all subcomponents to be created during
   initialization.
 - Remove healthcare access component.
 - Restructure mortality calculation in a style more consistent with
   risk-disease pairs.
 - Update to new API for simulation creation.
 - Remove usages of 'omit_missing_columns' in favor of population subviews.
 - Be consistent about rate naming conventions.
 - Rename Disability component to DisabilityObserver.
 - Rename 'age_group_start' and 'age_group_end' to 'age_start' and 'age_end'
   in data and lookup table usage.
 - Have components specify all necessary dependencies for the resources
   (pipelines, state table columns, and randomness streams) that they manage.
 - Update risk effect to make it easier to extend.
 - Allow lookup table specification without naming bin columns in data.
 - Update joint_value_postprocessor to union_postprocessor
 - Clean up some of the MSLT calculations
 - Dichotomous distribution bugfix

**0.9.19 - 09/30/19**

 - Add python and vivarium to the intersphinx mapping.
 - Bring in docs for non-standard risks.
 - Bugfix in parameterized risk component.
 - Update MSLT code to appropriate names/data artifact usage.

**0.9.18 - 07/29/19**

 - Pin pandas version to be compatible with tables.
 - Fix in RiskAttributableDisease disability calculation.

**0.9.17 - 07/17/19**

 - Add names to mslt components.
 - Clip non-ensemble distribution percentiles.

**0.9.16 - 07/16/19**

 - Update observers to not report ages younger than those modeled.

**0.9.15 - 07/03/19**

 - Fix docstring formatting.

**0.9.14 - 07/03/19**

 - Update api documentation format.
 - Bring in MSLT components.

**0.9.13 - 06/18/19**

 - Move ``VivariumError`` to the correct place.
 - Add names to all public health components.
 - Add several missing ``__repr__``s.
 - Modify the artifact to accept data that is wide on draws.
 - Update components to new component manager api.
 - Bugfix in SimulationDistribution

**0.9.12 - 04/23/19**

 - Update docstring for categorical risk observer.
 - Fix pipeline names in risk attributable disease.

**0.9.11 - 04/22/19**

 - Add documentation for the data artifact.
 - Bugfix in parameterized risk for covariates.
 - Make disease observers work with paf of one risks.
 - Make mortality and disability observers work with risk attributable diseases.
 - Add simulation info to simulant creator.

**0.9.10 - 03/29/19**

 - Bugfix in disease observer.

**0.9.9 - 03/28/19**

 - Bugfix in data free risk components when using a covariate for coverage.
 - Bugfix for simulations that start in a future year with extrapolate.

**0.9.8 - 03/19/19**

 - Bugfix in mortality observer.

**0.9.7 - 03/17/19**

 - Bugfixes in disease and treatment observers.
 - Remove unnecessary output metrics.

**0.9.6 - 03/13/19**

 - Generic observers for mortality, disability, person time, and treatment counts.
 - Bugfix for large propensities when using risk distributions.
 - Bugfix for rr distribution parameter name.

**0.9.5 - 03/01/19**

 - Bugfix in validating rebinning risks for continuous risks.

**0.9.4 - 03/01/19**

 - Added neonatal models and support for birth prevalence in DiseaseModel.
 - Added a risk attributable disease model.
 - Added support for rebinning polytomous risks into dichotomous risks.

**0.9.3 - 02/26/19**

 - Bugfix in checking relative risk source type from configuration.

**0.9.2 - 02/22/19**

 - Pin numpy and tables dependencies.
 - Remove forecast flags
 - Update crude birth rate fertility component
 - Allow parameterization of RiskEffect components with normal and lognormal distributions.
 - New observers for disease and treatment.

**0.9.1 - 02/14/19**

 - Update dependencies

**0.9.0 - 02/12/19**

 - Dataset manager logging.
 - Added an SIR with duration model.
 - Built observer for death counts and person years by age and year.
 - Updated population and crude birth rate models for GBD 2017.
 - Built an observer to point sample categorical risk exposure.
 - Updated risk distribution and effect to work with the updated risk_distributions package.
 - Updated healthcare access component.
 - Added component for therapeutic inertia.
 - Exposed individual cause disability weights as pipelines.
 - Various bugfixes and api updates.

**0.8.13 - 01/04/19**

 - Added support for multi-location data artifacts.
 - Added CI branch synchronization

**0.8.12 - 12/27/18**

 - Bugfix in categorical paf calculation

**0.8.11 - 12/20/18**

 - Bugfix for mock_artifact testing data to include newly added columns.
 - Bugfix to handle single-value sequela disability weight data.

**0.8.10 - 12/20/18**

 - Added a replace function to the artifact class.
 - Fixed a bug in age-specific fertility rate component.
 - Added data free risk and risk effect components
 - Removed the autogeneration of risk effects.
 - Updated the risk and risk effect API.
 - Added a configuration flag and component updates for limited forecasting data usage.
 - Put in cause-level disability weights.
 - Updated the population API.
 - Added in standard epi disease models.
 - Added support for morbidity only diseases.
 - Expanded risk effects to target excess mortality.
 - A host of model fixes and updates for the MSLT subpackage.

**0.8.9 - 11/15/18**

 - Update documentation dependencies.

**0.8.8 - 11/15/18**

 - Fix bug in population age generation.
 - Assign initial event time for prevalent cases of disease with a dwell time.
 - Set up artifact filter terms.
 - Remove mean year and age columns.

**0.8.7 - 11/07/18**

 - Switch to calculating pafs on the fly for non-continuous risks.
 - Adding components for mslt.
 - Pulled out distributions into separate package.

**0.8.6 - 11/05/18**

 - Extend interactive api to package up data artifact manager in standard sims.
 - Exposed disease prevalence propensity as a pipeline
 - Added logic to rebin polytomous risks to dichotomous risks.
 - Cleaned up confusing naming in metrics pipelines.
 - Allow open cohorts to extrapolate birth rate data into the future.

**0.8.5 - 10/23/18**

 - Update mass treatment campaign configuration for easier distributed runs.
 - Fix leaking global state in mock artifact.
 - Correctly implement order 0 interpolation.

**0.8.4 - 10/09/18**

 - Fix bug that caused dead people to still experience disease transitions.
 - Switch risk components to use pipelines for exposure/propensity
 - Cleaned up return types from distribution.ppf
 - Added indirect effects

**0.8.3 - 09/27/18**

 - Remove caching from artifact writes as it causes bugs.

**0.8.2 - 09/05/18**

 - Fix bug where the artifact manager assumed the data to be dataframe
 - Fix bug where the hdf applied filters even where it is not valid.

**0.8.1 - 08/22/18**

 - Fix various deployment things
 - Add badges
 - Remove unused metrics components
 - Use __about__ in docs
 - Extracted `Artifact` as an abstraction over hdf files.
 - Cleaned up Artifact manager plugin
 - Updated mock artifact

**0.8.0 - 07/24/18**

 - Initial Release
