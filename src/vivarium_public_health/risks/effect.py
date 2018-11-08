import numpy as np

from vivarium_public_health.util import pivot_age_sex_year_binned
from .data_transformation import should_rebin, rebin_rr_data, get_paf_data


def get_exposure_effect(builder, risk, risk_type):
    distribution = builder.data.load(f'{risk_type}.{risk}.distribution')
    risk_exposure = builder.value.get_value(f'{risk}.exposure')

    if distribution in ['normal', 'lognormal', 'ensemble']:
        tmred = builder.data.load(f"{risk_type}.{risk}.tmred")
        tmrel = 0.5 * (tmred["min"] + tmred["max"])
        exposure_parameters = builder.data.load(f"{risk_type}.{risk}.exposure_parameters")
        max_exposure = exposure_parameters["max_rr"]
        scale = exposure_parameters["scale"]

        def exposure_effect(rates, rr):
            exposure = np.minimum(risk_exposure(rr.index), max_exposure)
            relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
            return rates * relative_risk
    else:

        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            return rates * (rr.lookup(exposure.index, exposure))

    return exposure_effect


class RiskEffect:

    def __init__(self, risk, affected_entity, risk_type, affected_entity_type, get_data_functions=None):
        self.risk = risk
        self.risk_type = risk_type
        self.affected_entity = affected_entity
        self.affected_entity_type = affected_entity_type
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    @property
    def target(self):
        raise NotImplementedError()

    def setup(self, builder):
        paf_data = self._get_paf_data(builder)
        rr_data = self._get_rr_data(builder)
        self.population_attributable_fraction = builder.lookup.build_table(paf_data)
        self.relative_risk = builder.lookup.build_table(rr_data)

        self.exposure_effect = get_exposure_effect(builder, self.risk, self.risk_type)

        builder.value.register_value_modifier(f'{self.affected_entity}.{self.target}', modifier=self.adjust_target)
        builder.value.register_value_modifier(f'{self.affected_entity}.paf',
                                              modifier=self.population_attributable_fraction)

    def adjust_target(self, index, target):
        return self.exposure_effect(target, self.relative_risk(index))

    def _get_paf_data(self, builder):
        filter_name, filter_term = self.affected_entity_type, self.affected_entity
        if 'paf' in self._get_data_functions:
            paf_data = self._get_data_functions['paf'](builder)

        else:
            distribution = builder.data.load(f'{self.risk_type}.{self.risk}.distribution')
            if distribution in ['normal', 'lognormal', 'ensemble']:
                paf_data = builder.data.load(f'{self.risk_type}.{self.risk}.population_attributable_fraction')

            else:
                exposure = builder.data.load(f'{self.risk_type}.{self.risk}.exposure')
                rr = builder.data.load(f'{self.risk_type}.{self.risk}.relative_risk')
                rr = rr[rr[filter_name] == filter_term]
                paf_data = get_paf_data(exposure, rr)

        paf_data = paf_data[paf_data[filter_name] == filter_term]
        paf_data = paf_data.loc[:, ['sex', 'value', self.affected_entity_type, 'age_group_start', 'age_group_end',
                                    'year_start', 'year_end']]

        return pivot_age_sex_year_binned(paf_data, self.affected_entity_type, 'value')

    def _get_rr_data(self, builder):
        if 'rr' in self._get_data_functions:
            rr_data = self._get_data_functions['rr'](builder)
        else:
            rr_data = builder.data.load(f"{self.risk_type}.{self.risk}.relative_risk")

        row_filter = rr_data[f'{self.affected_entity_type}'] == self.affected_entity
        column_filter = ['parameter', 'sex', 'value', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
        rr_data = rr_data.loc[row_filter, column_filter]

        if should_rebin(self.risk, builder.configuration):
            exposure_data = builder.data.load(f"{self.risk_type}.{self.risk}.exposure")
            exposure_data = exposure_data.loc[:, column_filter]
            exposure_data = exposure_data[exposure_data['year_start'].isin(rr_data.year_start.unique())]
            rr_data = rebin_rr_data(rr_data, exposure_data)

        return pivot_age_sex_year_binned(rr_data, 'parameter', 'value')


class DirectEffect(RiskEffect):

    @property
    def target(self):
        return 'incidence_rate'


class IndirectEffect(RiskEffect):
    @property
    def target(self):
        return 'exposure_parameters'


class RiskEffectSet:
    def __init__(self, risk, risk_type):
        self.risk = risk
        self.risk_type = risk_type

    def setup(self, builder):
        affected_causes = builder.data.load(f"{self.risk_type}.{self.risk}.affected_causes")
        affected_risks = builder.data.load(f"{self.risk_type}.{self.risk}.affected_risk_factors")

        direct_effects = [
            DirectEffect(self.risk, cause, self.risk_type, 'cause') for cause in affected_causes
        ]
        indirect_effects = [
            IndirectEffect(self.risk, affected_risk, self.risk_type, 'risk_factor') for affected_risk in affected_risks
        ]

        builder.components.add_components(direct_effects + indirect_effects)
