import pandas as pd

from .state import DiseaseState, ExcessMortalityState


def make_disease_state(cause, side_effect_function=None):
    prevalence = cause.prevalence()
    disability_weight = cause.disability_weight()
    excess_mortality = cause.excess_mortality()
    dwell_time = cause.duration()

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

