"""
==============================
Data Transformations (Legacy)
==============================

.. deprecated:: 4.3.0
   This module is deprecated. Use :mod:`vivarium_public_health.exposure.data_transformations` instead.

Backward compatibility module for risk data_transformations.

This module provides backward compatibility for imports that expect risk
distribution classes to be in vivarium_public_health.risks.data_transformations.

The actual distribution classes have been moved to:
vivarium_public_health.exposure.data_transformations

This module will be deprecated in a future version.
"""

import warnings

# Issue a deprecation warning when this module is imported
warnings.warn(
    "Importing from 'vivarium_public_health.risks.data_transformations' is deprecated. "
    "Please import from 'vivarium_public_health.exposure.data_transformations' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all the classes from the new location
from vivarium_public_health.exposure.data_transformations import *
from vivarium_public_health.exposure.data_transformations import _rebin_relative_risk_data
