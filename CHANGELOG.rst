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
