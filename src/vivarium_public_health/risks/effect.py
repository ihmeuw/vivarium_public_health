import numpy as np

from vivarium_public_health.util import pivot_age_sex_year_binned
from .data_transformation import (should_rebin, rebin_rr_data, get_paf_data, exposure_from_covariate,
                                  build_exp_data_from_config, exposure_rr_from_config_value)


class RiskEffect:

    def __init__(self, risk_type, risk, affected_entity_type, affected_entity, target, get_data_functions=None):
        self.risk = risk
        self.risk_type = risk_type
        self.affected_entity = affected_entity
        self.affected_entity_type = affected_entity_type
        self.target = target
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        paf_data = self._get_paf_data(builder)
        rr_data = self._get_rr_data(builder)
        self.population_attributable_fraction = builder.lookup.build_table(paf_data)
        self.relative_risk = builder.lookup.build_table(rr_data)

        self.exposure_effect = self.get_exposure_effect(builder, self.risk, self.risk_type)

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

    @staticmethod
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


class RiskEffectSet:
    def __init__(self, risk, risk_type):
        self.risk = risk
        self.risk_type = risk_type

    def setup(self, builder):
        affected_causes = builder.data.load(f"{self.risk_type}.{self.risk}.affected_causes")
        affected_risks = builder.data.load(f"{self.risk_type}.{self.risk}.affected_risk_factors")

        direct_effects = [
            RiskEffect(self.risk_type, self.risk, 'cause', cause, 'incidence_rate') for cause in affected_causes
        ]
        indirect_effects = [
            RiskEffect(self.risk_type, self.risk, 'risk_factor', affected_risk, 'exposure_parameters') for affected_risk in affected_risks
        ]

        builder.components.add_components(direct_effects + indirect_effects)


class DummyRiskEffect(RiskEffect):

    configuration_defaults = {
        'effect_of_risk_on_entity': {
            'incidence_rate': 2,
            'exposure_parameters': 2,
        }
    }

    # TODO: do we want to allow get_data_functions for dummyriskeffect?
    def __init__(self, risk_type, risk, affected_entity_type, affected_entity, target):
        super().__init__(risk_type, risk, affected_entity_type, affected_entity, target)
        self.configuration_defaults = {f'effect_of_{self.risk}_on_{self.affected_entity}':
                                       DummyRiskEffect.configuration_defaults['effect_of_risk_on_entity']}

    def _get_paf_data(self, builder):
        exposure = build_exp_data_from_config(builder, self.risk)

        rr = self._build_rr_data_from_config(builder)

        paf_data = get_paf_data(exposure, rr)

        paf_data = paf_data.loc[:, ['sex', 'value', self.affected_entity_type, 'age_group_start', 'age_group_end',
                                    'year_start', 'year_end']]

        return pivot_age_sex_year_binned(paf_data, self.affected_entity_type, 'value')

    def _build_rr_data_from_config(self, builder):
        rr_config_key = f'effect_of_{self.risk}_on_{self.affected_entity}'
        rr_value = builder.configuration[rr_config_key][self.target]

        if not isinstance(rr_value, (int, float)):
            raise TypeError(f"You may only specify a single numeric value for relative risk of {rr_config_key} "
                            f"in the configuration. You supplied {rr_value}.")
        if rr_value < 1 or rr_value > 100:
            raise ValueError(f"The specified value for {rr_config_key} should be in the range [1, 100]. "
                             f"You specified {rr_value}")

        rr_data = exposure_rr_from_config_value(rr_value, builder.configuration.time.start.year,
                                                builder.configuration.time.end.year, 'relative_risk')
        return rr_data

    def _get_rr_data(self, builder):
        return pivot_age_sex_year_binned(self._build_rr_data_from_config(builder), 'parameter', 'value')

    @staticmethod
    def get_exposure_effect(builder, risk, risk_type):
        risk_exposure = build_exp_data_from_config(builder, risk)

        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            return rates * (rr.lookup(exposure.index, exposure))

        return exposure_effect

