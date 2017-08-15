import numbers

import pandas as pd

from ceam_inputs import get_disability_weight, get_prevalence, get_excess_mortality, get_duration

from .state import DiseaseState, ExcessMortalityState


def make_disease_state(cause, side_effect_function=None):
    prevalence = get_prevalence(cause)
    disability_weight = get_disability_weight(cause)
    excess_mortality = get_excess_mortality(cause)
    dwell_time = get_duration(cause)

    if isinstance(excess_mortality, pd.DataFrame) or excess_mortality > 0:
        return ExcessMortalityState(cause.name,
                                    prevalence_data=prevalence,
                                    dwell_time=dwell_time,
                                    disability_weight=disability_weight,
                                    excess_mortality_data=excess_mortality,
                                    side_effect_function=side_effect_function)
    return DiseaseState(cause.name,
                        prevalence_data=prevalence,
                        dwell_time=dwell_time,
                        disability_weight=disability_weight,
                        side_effect_function=side_effect_function)

