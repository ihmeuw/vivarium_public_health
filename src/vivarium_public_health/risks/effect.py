import numpy as np

from vivarium_public_health.util import pivot_age_sex_year_binned
from .data_transformation import (should_rebin, rebin_rr_data, get_paf_data, build_exp_data_from_config,
                                  build_rr_data_from_config, split_risk_from_type, split_target_from_type_entity)


class RiskEffect:
    """A component to model the impact of a risk factor on the target rate of
    some affected entity.

        Attributes
        ----------
        risk_type :
            'risk_factor' or 'coverage_gap'
        risk :
            The name of the risk factor
        affected_entity_type :
            The type of the entity affected by the risk factor, e.g., 'cause'
        affected_entity :
            The name of the entity affected by the risk factor

        """
    def __init__(self, risk: str, target: str, get_data_functions: dict=None):
        """

        Parameters
        ----------
        risk :
            Type and name of risk factor, supplied in the form
            "risk_type.risk_name" where risk_type should be singular (e.g.,
            risk_factor instead of risk_factors).
        target :
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        get_data_functions :
            Optional mapping of measure name to function to retrieve paf and rr
            data instead of reading from builder.data.

        """
        self.risk_type, self.risk = split_risk_from_type(risk)
        self.affected_entity_type, self.affected_entity, self.target = split_target_from_type_entity(target)
        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    def setup(self, builder):
        paf_data = self._get_paf_data(builder)
        rr_data = self._get_rr_data(builder)
        self.population_attributable_fraction = builder.lookup.build_table(paf_data)
        self.relative_risk = builder.lookup.build_table(rr_data)

        self.exposure_effect = self.get_exposure_effect(builder, self.risk, self.risk_type)

        builder.value.register_value_modifier(f'{self.affected_entity}.{self.target}', modifier=self.adjust_target)
        builder.value.register_value_modifier(f'{self.affected_entity}.{self.target}.paf',
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


class DummyRiskEffect(RiskEffect):
    """A component to model the impact of a risk factor on the target rate of
    some affected entity based only on data supplied in the model configuration.

    For a risk factor name 'dummy_risk' that affects the exposure of a risk
    named 'affected_risk', the configuration would look like:

    configuration:
        effect_of_dummy_risk_on_affected_risk:
            exposure_parameters: 2

        Attributes
        ----------
        risk_type :
            'risk_factor' or 'coverage_gap'
        risk :
            The name of the risk factor
        affected_entity_type :
            The type of the entity affected by the risk factor, e.g., 'cause'
        affected_entity :
            The name of the entity affected by the risk factor
        """

    configuration_defaults = {
        'effect_of_risk_on_entity': {
            'incidence_rate': 2,
            'exposure_parameters': 2,
        }
    }

    def __init__(self, risk: str, target: str):
        super().__init__(risk, target)
        self.configuration_defaults = {f'effect_of_{self.risk}_on_{self.affected_entity}':
                                       DummyRiskEffect.configuration_defaults['effect_of_risk_on_entity']}

    def _get_paf_data(self, builder):
        exposure = build_exp_data_from_config(builder, self.risk)

        rr = build_rr_data_from_config(builder, self.risk, self.affected_entity, self.target)
        rr[self.affected_entity_type] = self.affected_entity

        paf_data = get_paf_data(exposure, rr)

        paf_data = paf_data.loc[:, ['sex', 'value', self.affected_entity_type, 'age_group_start', 'age_group_end',
                                    'year_start', 'year_end']]

        return pivot_age_sex_year_binned(paf_data, self.affected_entity_type, 'value')

    def _get_rr_data(self, builder):
        return pivot_age_sex_year_binned(build_rr_data_from_config(builder, self.risk, self.affected_entity,
                                                                   self.target), 'parameter', 'value')

    @staticmethod
    def get_exposure_effect(builder, risk, risk_type):
        risk_exposure = builder.value.get_value(f'{risk}.exposure')

        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            return rates * (rr.lookup(exposure.index, exposure))

        return exposure_effect
