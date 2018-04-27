import numpy as np
import pandas as pd


def continuous_exposure_effect(risk, population_view):
    """Factory that makes functions which can be used as the exposure_effect for standard continuous risks.

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk+'_exposure'
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
    exposure_column = risk+'_exposure'

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

    def __init__(self, risk, cause, get_data_functions=None, risk_type="risk_factor"):
        self.risk = risk
        self.risk_type = risk_type
        self.cause = cause
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        paf_data = self._get_data_functions.get('paf', lambda risk, cause, builder: builder.data.load(f"{self.risk_type}.{risk}.population_attributable_fraction", cause=self.cause))(self.risk, self.cause, builder)
        self.population_attributable_fraction = builder.lookup(paf_data)
        if paf_data.empty:
            #FIXME: Bailing out because we don't have preloaded data for this cause-risk pair. This should be handled higher up but since it isn't yet I'm just going to skip all the plumbing leaving this as a NOP component
            return

        self._raw_rr =  self._get_data_functions.get('rr', lambda risk, cause, builder: builder.data.load(f"{self.risk_type}.{risk}.relative_risk", cause=self.cause))(self.risk, self.cause, builder)

        # FIXME: Find a better way of defering this
        self.__lookup_builder_function = builder.lookup
        self._relative_risk = None


        if builder.configuration.risks.apply_mediation:
            mf =  self._get_data_functions.get('mf', lambda risk, cause, builder: builder.data.load(f"{self.risk_type}.{risk}.mediation_factor", cause=self.cause))(self.risk, self.cause, builder)
            if mf is not None and not mf.empty:
                self.mediation_factor = builder.lookup(float(mf.value))
            else:
                self.mediation_factor = None
        else:
            self.mediation_factor = None

        builder.value.register_value_modifier(f'{self.cause}.incidence_rate', modifier=self.incidence_rates)
        builder.value.register_value_modifier(f'{self.cause}.paf', modifier=self.paf_mf_adjustment)
        self.population_view = builder.population.get_view([self.risk + '_exposure'])
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


class RiskEffectSet:
    def __init__(self, risk, risk_type="risk_factor"):
        self.risk = risk
        self.risk_type = risk_type

    def setup(self, builder):
        return [RiskEffect(risk=self.risk, cause=cause, risk_type=self.risk_type) for cause in builder.data.load(f"{self.risk_type}.{self.risk}.affected_causes")]
