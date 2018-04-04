import numpy as np
import pandas as pd


def continuous_exposure_effect(risk, population_view):
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
    def inner(rates, rr):
        exposure = np.minimum(population_view.get(rr.index)[exposure_column].values, max_exposure)
        relative_risk = np.maximum(rr.values**((exposure - tmrel) / scale), 1)
        return rates * relative_risk

    return inner


def categorical_exposure_effect(risk, population_view):
    """Factory that makes functions which can be used as the exposure_effect for binary categorical risks

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'

    def inner(rates, rr):
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

    def setup(self, builder):
        self._raw_rr =  self._get_data_functions.get('rr', lambda risk, cause, builder: builder.data.load(f"risk_factor.{risk.name}.relative_risk", cause_id=self.cause.gbd_id))(self.risk, self.cause, builder)

        # FIXME: Find a better way of defering this
        self.__lookup_builder_function = builder.lookup
        self._relative_risk = None

        paf_data = self._get_data_functions.get('paf', lambda risk, cause, builder: builder.data.load(f"risk_factor.{risk.name}.population_attributable_fraction", cause_id=self.cause.gbd_id))(self.risk, self.cause, builder)
        self.population_attributable_fraction = builder.lookup(paf_data)

        if builder.configuration.risks.apply_mediation:
            mf =  self._get_data_functions.get('mf', lambda risk, cause, builder: builder.data.load(f"risk_factor.{risk.name}.mediation_factor", cause_id=self.cause.gbd_id))(self.risk, self.cause, builder)
            if mf is not None and not mf.empty:
                self.mediation_factor = builder.lookup(mf)
            else:
                self.mediation_factor = None
        else:
            self.mediation_factor = None

        builder.value.register_value_modifier(f'{self.cause_name}.incidence_rate', modifier=self.incidence_rates)
        builder.value.register_value_modifier(f'{self.cause_name}.paf', modifier=self.paf_mf_adjustment)
        self.population_view = builder.population.get_view([self.risk.name + '_exposure'])
        is_continuous = self.risk.distribution in ['lognormal', 'ensemble', 'normal']
        self.exposure_effect = (continuous_exposure_effect(self.risk, self.population_view) if is_continuous
                                else categorical_exposure_effect(self.risk, self.population_view))


    @property
    def relative_risk(self):
        if self._relative_risk is None:
            if self.risk.distribution in ('dichotomous', 'polytomous'):
                # TODO: I'm not sure this is the right place to be doing this reshaping. Maybe it should
                # be in the data_transformations somewhere?
                rr_data = pd.pivot_table(self._raw_rr, index=['year', 'age', 'sex'],
                                               columns='parameter', values='relative_risk').dropna()
                rr_data = self.__rr_data.reset_index()
            else:
                rr_data = self._raw_rr
            self._relative_risk = self.__lookup_builder_function(rr_data)
        return self._relative_risk


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
