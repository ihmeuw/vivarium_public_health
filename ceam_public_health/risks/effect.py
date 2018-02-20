import numpy as np
import pandas as pd

from ceam_inputs import get_relative_risk, get_population_attributable_fraction, get_mediation_factor

from vivarium.framework.population import uses_columns


def continuous_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for standard continuous risks.

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'
    tmrel = 0.5 * (risk.tmred.min + risk.tmred.max)
    max_exposure = risk.exposure_parameters.max_rr
    scale = risk.exposure_parameters.scale

    # FIXME: Exposure, TMRL, and Scale values should be part of the values pipeline system.
    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        exposure = np.minimum(population_view.get(rr.index)[exposure_column].values, max_exposure)
        relative_risk = np.maximum(rr.values**((exposure - tmrel) / scale), 1)
        return rates * relative_risk

    return inner


def categorical_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for binary categorical risks

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
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
    """

    configuration_defaults = {
        'risks': {
            'apply_mediation': True,
        },
    }

    def __init__(self, risk, cause, get_data_functions=None):
        self.risk = risk
        self.cause = cause
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

        self.cause_name = cause.name

        is_continuous = self.risk.distribution in ['lognormal', 'ensemble', 'normal']
        self.exposure_effect = (continuous_exposure_effect(self.risk) if is_continuous
                                else categorical_exposure_effect(self.risk))

    def setup(self, builder):
        get_rr_func = self._get_data_functions.get('rr', get_relative_risk)
        get_paf_func = self._get_data_functions.get('paf', get_population_attributable_fraction)
        get_mf_func = self._get_data_functions.get('mf', get_mediation_factor)

        self._rr_data = get_rr_func(self.risk, self.cause, builder.configuration)

        if self.risk.distribution in ('dichotomous', 'polytomous'):
            # TODO: I'm not sure this is the right place to be doing this reshaping. Maybe it should
            # be in the data_transformations somewhere?
            self._rr_data = pd.pivot_table(self._rr_data, index=['year', 'age', 'sex'],
                                           columns='parameter', values='relative_risk').dropna()
            self._rr_data = self._rr_data.reset_index()
        else:
            del self._rr_data['parameter']

        self._paf_data = get_paf_func(self.risk, self.cause, builder.configuration)
        self._mediation_factor = get_mf_func(self.risk, self.cause, builder.configuration)

        self.relative_risk = builder.lookup(self._rr_data)
        self.population_attributable_fraction = builder.lookup(self._paf_data)

        if builder.configuration.risks.apply_mediation:
            self.mediation_factor = builder.lookup(self._mediation_factor)
        else:
            self.mediation_factor = None

        builder.value.register_value_modifier(f'{self.cause_name}.incidence_rate', modifier=self.incidence_rates)
        builder.value.register_value_modifier(f'{self.cause_name}.paf', modifier=self.paf_mf_adjustment)

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
        return "RiskEffect(cause= {})".format(self.cause)


def make_gbd_risk_effects(risk):
    return [RiskEffect(risk=risk, cause=cause) for cause in risk.affected_causes]
