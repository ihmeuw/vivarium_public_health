import numpy as np
import pandas as pd


def continuous_exposure_effect(risk, risk_type, population_view, builder):
    """Factory that makes functions which can be used as the exposure_effect for standard continuous risks.

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk+'_exposure'
    tmred = builder.data.load(f"{risk_type}.{risk}.tmred")
    tmrel = 0.5 * (tmred["min"] + tmred["max"])
    exposure_parameters = builder.data.load(f"{risk_type}.{risk}.exposure_parameters")
    max_exposure = exposure_parameters["max_rr"]
    scale = exposure_parameters["scale"]

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

    def __init__(self, risk, cause, get_data_functions=None, risk_type="risk_factor", cause_type="cause"):
        self.risk = risk
        self.risk_type = risk_type
        self.cause = cause
        self.cause_type = cause_type
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        #TODO Handle various types better than this. Maybe different kinds of RiskEffect?
        if self.risk_type == "risk_factor":
            paf_data = self._get_data_functions.get('paf', lambda risk, cause, builder: builder.data.load(
                f"{self.cause_type}.{cause}.population_attributable_fraction", risk=risk))(self.risk, self.cause, builder)
        else:
            paf_data = self._get_data_functions.get('paf', lambda risk, cause, builder: builder.data.load(
                f"{self.risk_type}.{risk}.population_attributable_fraction", cause=cause))(self.risk, self.cause, builder)

        self.population_attributable_fraction = builder.lookup.build_table(paf_data[['year', 'sex', 'age', 'value']])
        if paf_data.empty:
            #FIXME: Bailing out because we don't have preloaded data for this cause-risk pair.
            # This should be handled higher up but since it isn't yet I'm just going to
            # skip all the plumbing leaving this as a NOP component
            return

        rr_data = self._get_data_functions.get('rr', lambda risk, cause, builder: builder.data.load(
            f"{self.risk_type}.{risk}.relative_risk", cause=self.cause))(self.risk, self.cause, builder)[
            ['year', 'parameter', 'sex', 'age', 'value']]

        rr_data = pd.pivot_table(rr_data, index=['year', 'age', 'sex'],
                                 columns='parameter', values='value').dropna()
        rr_data = rr_data.reset_index()
        self.relative_risk = builder.lookup.build_table(rr_data)

        builder.value.register_value_modifier(f'{self.cause}.incidence_rate', modifier=self.incidence_rates)
        builder.value.register_value_modifier(f'{self.cause}.paf',
                                              modifier=self.population_attributable_fraction)

        self.population_view = builder.population.get_view([self.risk + '_exposure'])
        distribution = builder.data.load(f"{self.risk_type}.{self.risk}.distribution")
        self.is_continuous = distribution in ['lognormal', 'ensemble', 'normal']
        self.exposure_effect = (continuous_exposure_effect(self.risk, self.risk_type, self.population_view, builder)
                                if self.is_continuous else categorical_exposure_effect(self.risk, self.population_view))

    def incidence_rates(self, index, rates):
        return self.exposure_effect(rates, self.relative_risk(index))

    def __repr__(self):
        return f"RiskEffect(risk={self.risk}, cause={self.cause})"


class RiskEffectSet:
    def __init__(self, risk, risk_type="risk_factor"):
        self.risk = risk
        self.risk_type = risk_type

    def setup(self, builder):
        builder.components.add_components([RiskEffect(risk=self.risk, cause=cause, risk_type=self.risk_type) for cause
                                           in builder.data.load(f"{self.risk_type}.{self.risk}.affected_causes")])
