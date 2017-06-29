import numpy as np

from ceam import config

import ceam_inputs as inputs
from ceam.framework.population import uses_columns


def continuous_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for standard continuous risks.

    Parameters
    ----------
    risk : `ceam.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'

    # FIXME: Exposure, TMRL, and Scale values should be part of the values pipeline system.
    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        return rates * np.maximum(
            rr.values**((population_view.get(rr.index)[exposure_column] - risk.tmrl) / risk.scale).values, 1)

    return inner


def categorical_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for binary categorical risks

    Parameters
    ----------
    risk : `ceam.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'

    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        exposure_ = population_view.get(rr.index)[exposure_column]
        return rates * (rr.lookup(exposure_.index, exposure_))
    return inner


class RiskEffect:
    """RiskEffect objects bundle all the effects that a given risk has on a cause.
    
    Parameters
    ----------
    rr_data : pandas.DataFrame
        A dataframe of relative risk data with age, sex, year, and rr columns
    paf_data : pandas.DataFrame
        A dataframe of population attributable fraction data with age, sex, year, and paf columns
    cause : `ceam.config_tree.ConfigTree`
        The gbd data mapping for the cause.
    exposure_effect : callable
        A function which takes a series of incidence rates and a series of
        relative risks and returns rates modified as appropriate for this risk
    """

    configuration_defaults = {
        'risks': {
            'apply_mediation': True,
        },
    }

    def __init__(self, rr_data, paf_data, mediation_factor, cause, exposure_effect):
        self._rr_data = rr_data
        self._paf_data = paf_data
        self._mediation_factor = mediation_factor
        self.cause = cause
        self.exposure_effect = exposure_effect

    def setup(self, builder):
        self.relative_risk = builder.lookup(self._rr_data)
        self.population_attributable_fraction = builder.lookup(self._paf_data)

        if config.risks.apply_mediation:
            self.mediation_factor = builder.lookup(self._mediation_factor)
        else:
            self.mediation_factor = None
        builder.modifies_value(self.incidence_rates, 'incidence_rate.{}'.format(self.cause.name))
        builder.modifies_value(self.paf_mf_adjustment, 'paf.{}'.format(self.cause.name))

        return [self.exposure_effect]

    def paf_mf_adjustment(self, index):
        if self.mediation_factor:
            return self.population_attributable_fraction(index) * (1 - self.mediation_factor(index))
        else:
            return self.population_attributable_fraction(index)

    def incidence_rates(self, index, rates):
        if self.mediation_factor:
            return self.exposure_effect(rates, self.relative_risk(index).pow(1 - self.mediation_factor(index), axis=0))
        else:
            return self.exposure_effect(rates, self.relative_risk(index))


    def __repr__(self):
        return ("RiskEffect(rr_data= {},\npaf_data= {},\n".format(self._rr_data, self._paf_data)
                + "cause= {},\nexposure_effect= {},\n".format(self.cause.name, self.exposure_effect)
                + "mediation_factor= {})".format(self._mediation_factor))


def make_gbd_risk_effects(risk):
    effect_function = continuous_exposure_effect(risk) if risk.risk_type == 'continuous' else categorical_exposure_effect(risk)
    return [RiskEffect(rr_data=inputs.get_relative_risks(risk_id=risk.gbd_risk, cause_id=cause.gbd_cause),
                       paf_data=inputs.get_pafs(risk_id=risk.gbd_risk, cause_id=cause.gbd_cause),
                       mediation_factor=inputs.get_mediation_factors(risk_id=risk.gbd_risk, cause_id=cause.gbd_cause),
                       cause=cause,
                       exposure_effect=effect_function) for cause in risk.effected_causes]
