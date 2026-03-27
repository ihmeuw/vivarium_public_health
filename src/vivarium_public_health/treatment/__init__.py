"""
========================
Treatment Modeling Tools
========================

Provide components for modeling treatment interventions in public health
simulations.

This package includes tools for applying direct shifts to epidemiological
measures, linearly scaling up intervention coverage over time, modeling
therapeutic inertia, and defining intervention exposures with their
effects on target measures.

"""

from vivarium_public_health.treatment.intervention import Intervention, InterventionEffect
from vivarium_public_health.treatment.magic_wand import AbsoluteShift
from vivarium_public_health.treatment.scale_up import LinearScaleUp
from vivarium_public_health.treatment.therapeutic_inertia import TherapeuticInertia
