**0.9.3 - 02/26/19**

 - Bugfix in checking relative risk source type from configuration.

**0.9.2 - 02/22/19**

 - Pin numpy and tables dependencies.
 - Remove forecast flags
 - Update crude birth rate fertility component
 - Allow parameterization of RiskEffect components with normal and lognormal distributions.

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
