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
