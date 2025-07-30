"""
Backward compatibility module for risk distributions.

This module provides backward compatibility for imports that expect risk
distribution classes to be in vivarium_public_health.risks.distributions.

The actual distribution classes have been moved to:
vivarium_public_health.exposure.distributions

This module will be deprecated in a future version.
"""

import warnings

# Issue a deprecation warning when this module is imported
warnings.warn(
    "Importing from 'vivarium_public_health.risks.distributions' is deprecated. "
    "Please import from 'vivarium_public_health.exposure.distributions' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import all the classes from the new location
from vivarium_public_health.exposure.distributions import *
