from typing import Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness import RandomnessStream
from vivarium_public_health.risks import distributions


#############
# Utilities #
#############

class RiskString(str):
    """Convenience class for representing risks as strings."""

    def __init__(self, risk):
        super().__init__()
        self._type, self._name = self.split_risk()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    def split_risk(self):
        split = self.split('.')
        if len(split) != 2:
            raise ValueError(f'You must specify the risk as "risk_type.risk_name". You specified {self}.')
        return split[0], split[1]


class TargetString(str):
    """Convenience class for representing risk targets as strings."""

    def __init__(self, target):
        super().__init__()
        self._type, self._name, self._measure = self.split_target()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def measure(self):
        return self._measure

    def split_target(self):
        split = self.split('.')
        if len(split) != 3:
            raise ValueError(
                f'You must specify the target as "affected_entity_type.affected_entity_name.affected_measure".'
                f'You specified {self}.')
        return split[0], split[1], split[2]


def pivot_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """Pivots data that is long on categories to be wide."""
    key_cols = ['sex', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
    data = data.pivot_table(index=key_cols, columns='parameter', values='value').reset_index()
    data.columns.name = None
    return data


##########################
# Exposure data handlers #
##########################

def get_distribution(builder, risk: RiskString):
    validate_distribution_data_source(builder, risk)
    data = load_distribution_data(builder, risk)
    return distributions.get_distribution(risk.name, **data)


def get_exposure_post_processor(builder, risk: RiskString):
    thresholds = builder.configuration[risk.name]['category_thresholds']

    if thresholds:
        thresholds = [-np.inf] + thresholds + [np.inf]
        categories = [f'cat{i}' for i in range(1, len(thresholds))]

        def post_processor(exposure, _):
            return pd.Series(pd.cut(exposure, thresholds, labels=categories), index=exposure.index).astype(str)
    else:
        post_processor = None

    return post_processor


def load_distribution_data(builder, risk: RiskString):
    exposure_data = get_exposure_data(builder, risk)
    exposure_data = rebin_exposure_data(builder, risk, exposure_data)

    data = {'distribution_type': get_distribution_type(builder, risk),
            'exposure': exposure_data,
            'exposure_standard_deviation': get_exposure_standard_deviation_data(builder, risk),
            'weights': get_exposure_distribution_weights(builder, risk)}
    return data


def get_distribution_type(builder, risk: RiskString):
    risk_config = builder.configuration[risk.name]

    if risk_config['exposure'] == 'data':
        distribution_type = builder.data.load(f'{risk}.distribution')
    else:
        distribution_type = 'dichotomous'

    return distribution_type


def get_exposure_data(builder, risk: RiskString):
    risk_config = builder.configuration[risk.name]
    exposure_source = risk_config['exposure']
    distribution_type = get_distribution_type(builder, risk)

    if exposure_source == 'data':
        exposure_data = builder.data.load(f'{risk}.exposure')
    else:
        if isinstance(exposure_source, str):  # Build from covariate
            cat1 = builder.data.load(f'covariate.{exposure_source}.estimate')
            # TODO: Generate a draw.
            cat1 = cat1[cat1['parameter'] == 'mean_value']
            cat1['parameter'] = 'cat1'
        else:  # We have a numerical value
            cat1 = builder.data.load('population.demographic_dimensions')
            cat1['parameter'] = 'cat1'
            cat1['value'] = float(exposure_source)
        cat2 = cat1.copy()
        cat2['parameter'] = 'cat2'
        cat2['value'] = 1 - cat2['value']
        exposure_data = pd.concat([cat1, cat2], ignore_index=True)

    if distribution_type in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        exposure_data = pivot_categorical(exposure_data)

    return exposure_data


def get_exposure_standard_deviation_data(builder, risk: RiskString):
    distribution_type = get_distribution_type(builder, risk)
    if distribution_type in ['normal', 'lognormal', 'ensemble']:
        exposure_sd = builder.data.load(f'{risk}.exposure_standard_deviation')
    else:
        exposure_sd = None
    return exposure_sd


def get_exposure_distribution_weights(builder, risk: RiskString):
    distribution_type = get_distribution_type(builder, risk)
    if distribution_type == 'ensemble':
        weights = builder.data.load(f'{risk}.exposure_distribution_weights')
        weights = pivot_categorical(weights)
        if 'glnorm' in weights.columns:
            if np.any(weights['glnorm']):
                raise NotImplementedError('glnorm distribution is not supported')
            weights = weights.drop(columns='glnorm')
    else:
        weights = None
    return weights


def rebin_exposure_data(builder, risk: RiskString, data: pd.DataFrame):
    rebin = builder.configuration[risk.name]['rebin']
    # if 'polytomous' in distribution_type:
    #     rebin_unsupported = ['unsafe_water_source', 'low_birth_weight_and_short_gestation']
    #     if risk_config['rebin'] and risk.name in rebin_unsupported:
    #         raise NotImplementedError(f'{risk.name} cannot be rebinned.')
    #     elif risk_config['rebin'] and not raw:
    #         exposure_data = rebin_exposure_data(exposure_data)

    # unexposed = sorted([c for c in data['parameter'].unique() if 'cat' in c], key=lambda c: int(c[3:]))[-1]
    # middle_cats = set(data['parameter']) - {unexposed} - {'cat1'}
    # data['sex'] = data['sex'].astype(object)
    # df = data.groupby(['year_start', 'year_end', 'age_group_start', 'age_group_end', 'sex'], as_index=False)
    # assert np.allclose(df['value'].sum().value, 1)
    #
    # def rebin(g):
    #     g.reset_index(inplace=True)
    #     to_drop = g['parameter'].isin(middle_cats)
    #     g.drop(g[to_drop].index, inplace=True)
    #
    #     g.loc[g.parameter == 'cat1', 'value'] = 1 - g[g.parameter == unexposed].loc[:, 'value'].values
    #     return g
    #
    # df = df.apply(rebin).reset_index().replace(unexposed, 'cat2')
    if rebin:
        raise NotImplementedError()
    return data


###############################
# Relative risk data handlers #
###############################

def get_relative_risk_data(builder, risk: RiskString, target: TargetString, randomness: RandomnessStream):
    source_type = validate_relative_risk_data_source(builder, risk, target)
    relative_risk_data = load_relative_risk_data(builder, risk, target, source_type, randomness)
    relative_risk_data = rebin_relative_risk_data(builder, risk, relative_risk_data)
    return relative_risk_data


def load_relative_risk_data(builder, risk: RiskString, target: TargetString,
                            source_type: str, randomness: RandomnessStream):
    distribution_type = get_distribution_type(builder, risk)
    relative_risk_source = builder.configuration[f'effect_of_{risk.name}_on_{target.name}'][target.measure]

    if source_type == 'data':
        relative_risk_data = builder.data.load(f'{risk}.relative_risk')
        correct_target = ((relative_risk_data['affected_entity'] == target.name)
                          & (relative_risk_data['affected_measure'] == target.measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))
        if distribution_type in ['normal', 'lognormal', 'ensemble']:
            relative_risk_data = relative_risk_data.drop(['parameter'], 'columns')

    elif source_type == 'relative risk value':
        relative_risk_data = _make_relative_risk_data(builder, float(relative_risk_source['relative_risk']))

    else:  # distribution
        parameters = {k: v.get_value() for k, v in relative_risk_source.items() if v.get_value() is not None}
        random_state = np.random.RandomState(randomness.get_seed())
        cat1_value = generate_relative_risk_from_distribution(random_state, parameters)
        relative_risk_data = _make_relative_risk_data(builder, cat1_value)

    if distribution_type in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        relative_risk_data = pivot_categorical(relative_risk_data)

    return relative_risk_data


def generate_relative_risk_from_distribution(random_state: np.random.RandomState,
                                             parameters: dict) -> Union[float, pd.Series, np.ndarray]:
    first = pd.Series(list(parameters.values())[0])
    length = len(first)
    index = first.index

    for v in parameters.values():
        if length != len(pd.Series(v)) or not index.equals(pd.Series(v).index):
            raise ValueError('If specifying vectorized parameters, all parameters '
                             'must be the same length and have the same index.')

    if 'mean' in parameters:  # normal distribution
        rr_value = random_state.normal(parameters['mean'], parameters['se'])
    elif 'log_mean' in parameters:  # log distribution
        rr_value = np.exp(parameters['log_se'] * random_state.randn()
                          + parameters['log_mean'] + random_state.normal(0, parameters['tau_squared']))
    else:
        raise NotImplementedError(f'Only normal distributions (supplying mean and se) and log distributions '
                                  f'(supplying log_mean, log_se, and tau_squared) are currently supported.')

    rr_value = np.maximum(1, rr_value)

    return rr_value


def _make_relative_risk_data(builder, cat1_value: float) -> pd.DataFrame:
    cat1 = builder.data.load('population.demographic_dimensions')
    cat1['parameter'] = 'cat1'
    cat1['value'] = cat1_value
    cat2 = cat1.copy()
    cat2['parameter'] = 'cat2'
    cat2['value'] = 1
    return pd.concat([cat1, cat2], ignore_index=True)


def rebin_relative_risk_data(builder, risk: RiskString, relative_risk_data: pd.DataFrame) -> pd.DataFrame:
    """ When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
        For the exposed categories of relative risk (after rebinning) should be the weighted sum of relative risk
        of those categories where weights are relative proportions of exposure of those categories.
        For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
        for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
        (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
    """
    # if 'polytomous' in distribution_type:
    #     rebin_unsupported = ['unsafe_water_source', 'low_birth_weight_and_short_gestation']
    #     if risk_config['rebin'] and risk.name in rebin_unsupported:
    #         raise NotImplementedError(f'{risk.name} cannot be rebinned.')
    #     elif risk_config['rebin']:
    #         relative_risk_data = rebin_relative_risk_data(exposure_data)

    # df = exposure.merge(rr, on=['parameter', 'sex', 'age_group_start', 'age_group_end',
    #                             'year_start', 'year_end'])
    #
    # df = df.groupby(['year_start', 'year_end', 'age_group_start', 'age_group_end', 'sex'], as_index=False)
    #
    # unexposed = sorted([c for c in rr['parameter'].unique() if 'cat' in c], key=lambda c: int(c[3:]))[-1]
    # middle_cats = set(rr['parameter']) - {unexposed} - {'cat1'}
    # rr['sex'] = rr['sex'].astype(object)
    # exposure['sex'] = exposure['sex'].astype(object)
    #
    # def rebin_rr(g):
    #     # value_x = exposure, value_y = rr
    #     g['weighted_rr'] = g['value_x']*g['value_y']
    #     x = g['weighted_rr'].sum()-g.loc[g.parameter == unexposed, 'weighted_rr']
    #     x /= g['value_x'].sum()-g.loc[g.parameter == unexposed, 'value_x'].values
    #     to_drop = g['parameter'].isin(middle_cats)
    #     g.drop(g[to_drop].index, inplace=True)
    #     g.drop(['value_x', 'value_y', 'weighted_rr'], axis=1, inplace=True)
    #     g['value'] = x.iloc[0]
    #     g.loc[g.parameter == unexposed, 'value'] = 1.0
    #     g['value'].fillna(0, inplace=True)
    #     return g
    #
    # df = df.apply(rebin_rr).reset_index().loc[:, ['parameter', 'sex', 'value', 'age_group_start',
    #                                               'age_group_end', 'year_start', 'year_end']]
    # return df.replace(unexposed, 'cat2')
    rebin = builder.configuration[risk.name]['rebin']
    if rebin:
        raise NotImplementedError()
    return relative_risk_data


def get_exposure_effect(builder, risk: RiskString):
    distribution_type = get_distribution_type(builder, risk)
    risk_exposure = builder.value.get_value(f'{risk.name}.exposure')

    if distribution_type in ['normal', 'lognormal', 'ensemble']:
        tmred = builder.data.load(f"{risk}.tmred")
        tmrel = 0.5 * (tmred["min"] + tmred["max"])
        scale = builder.data.load(f"{risk}.relative_risk_scalar")

        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
            return rates * relative_risk
    else:
        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            return rates * (rr.lookup(exposure.index, exposure))

    return exposure_effect


##################################################
# Population attributable fraction data handlers #
##################################################

def get_population_attributable_fraction_data(builder, risk: RiskString,
                                              target: TargetString, randomness: RandomnessStream):
    exposure_source = builder.configuration[f'{risk.name}']['exposure']
    rr_source_type = validate_relative_risk_data_source(builder, risk, target)

    if exposure_source == 'data' and rr_source_type == 'data' and risk.type == 'risk_factor':
        paf_data = builder.data.load(f'{risk}.population_attributable_fraction')
        correct_target = ((paf_data['affected_entity'] == target.name)
                          & (paf_data['affected_measure'] == target.measure))
        paf_data = (paf_data[correct_target]
                    .drop(['affected_entity', 'affected_measure'], 'columns'))
    else:
        key_cols = ['sex', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
        exposure_data = get_exposure_data(builder, risk).set_index(key_cols)
        relative_risk_data = get_relative_risk_data(builder, risk, target, randomness).set_index(key_cols)
        mean_rr = (exposure_data * relative_risk_data).sum(axis=1)
        paf_data = ((mean_rr - 1)/mean_rr).reset_index().rename(columns={0: 'value'})
    return paf_data


# def _get_paf_data(self, builder):
#     risk_config = builder.configuration[self.risk.name]
#     exposure_source = risk_config['exposure']
#     rr_source = builder.configuration[f'effect_of_{self.risk.name}_on_{self.target.name}'][self.target.measure]
#
#     if exposure_source == 'data' and rr_source == 'data':
#         paf_data = builder.data.load(f'{self.risk}.population_attributable_fraction')
#     elif exposure_source == 'data':
#         rr = self._get_relative_risk_data(builder)
#
#
#     # if self._config_data:
#     #     exposure = build_exp_data_from_config(builder, self.risk)
#     #     rr = build_rr_data_from_config(builder, self.risk, self.affected_entity, self.affected_measure)
#     #     paf_data = None #get_paf_data(exposure, rr)
#     #     paf_data['affected_entity'] = self.affected_entity
#     else:
#         if 'paf' in self._get_data_functions:
#             paf_data = self._get_data_functions['paf'](builder)
#         else:
#             distribution = builder.data.load(f'{self.risk_type}.{self.risk}.distribution')
#             if distribution in ['normal', 'lognormal', 'ensemble']:
#                 paf_data = builder.data.load(f'{self.risk_type}.{self.risk}.population_attributable_fraction')
#                 paf_data = paf_data[paf_data['affected_measure'] == self.affected_measure]
#                 paf_data = paf_data[paf_data['affected_entity'] == self.affected_entity]
#             else:
#                 exposure = builder.data.load(f'{self.risk_type}.{self.risk}.exposure')
#                 rr = builder.data.load(f'{self.risk_type}.{self.risk}.relative_risk')
#                 rr = rr[rr['affected_measure'] == self.affected_measure].drop('affected_measure', 'columns')
#                 rr = rr[rr['affected_entity'] == 'affected_entity'].drop('affected_entity', 'columns')
#                 paf_data = None #get_paf_data(exposure, rr)
#
#                 paf_data['affected_entity'] = self.affected_entity
#
#     paf_data = paf_data.loc[:, ['sex', 'value', 'affected_entity', 'age_group_start', 'age_group_end',
#                                 'year_start', 'year_end']]
#
#     return pivot_categorical(paf_data)

# if should_rebin(self.risk, builder.configuration):
#     exposure_data = builder.data.load(f"{self.risk_type}.{self.risk}.exposure")
#     exposure_data = exposure_data.loc[:, column_filter]
#     exposure_data = exposure_data[exposure_data['year_start'].isin(rr_data.year_start.unique())]
#     rr_data = rebin_rr_data(rr_data, exposure_data)


##############
# Validators #
##############

def validate_distribution_data_source(builder, risk: RiskString):
    """Checks that the exposure distribution specification is valid."""
    exposure_type = builder.configuration[risk.name]['exposure']
    rebin = builder.configuration[risk.name]['rebin']
    category_thresholds = builder.configuration[risk.name]['category_thresholds']

    if risk.type == 'alternative_risk_factor':
        if exposure_type != 'data' or rebin:
            raise ValueError('Parameterized risk components are not available for alternative risks.')

        if not category_thresholds:
            raise ValueError('Must specify category thresholds to use alternative risks.')

    elif risk.type in ['risk_factor', 'coverage_gap']:
        if isinstance(exposure_type, (int, float)) and not 0 <= exposure_type <= 1:
            raise ValueError(f"Exposure should be in the range [0, 1]")
        elif isinstance(exposure_type, str) and exposure_type.split('.')[0] not in ['covariate', 'data']:
            raise ValueError(f"Exposure must be specified as 'data', an integer or float value, "
                             f"or as a string in the format covariate.covariate_name")
        else:
            pass  # All good
    else:
        raise ValueError(f'Unknown risk type {risk.type} for risk {risk.name}')


def validate_relative_risk_data_source(builder, risk: RiskString, target: TargetString):
    source_key = f'effect_of_{risk.name}_on_{target.name}'
    relative_risk_source = builder.configuration[source_key][target.measure]

    provided_keys = set(k for k, v in relative_risk_source.items() if isinstance(v.get_value(), (int, float)))

    source_map = {'data': set(),
                  'relative risk value': {'relative_risk'},
                  'normal distribution': {'mean', 'se'},
                  'log distribution': {'log_mean', 'log_se', 'tau_squared'}}

    if provided_keys not in source_map.values():
        raise ValueError(f'The acceptable parameter options for specifying relative risk are: '
                         f'{source_map.values()}. You provided {provided_keys} for {source_key}.')

    source_type = [k for k, v in source_map.items() if provided_keys == v][0]

    if source_type == 'relative risk value':
        if not 1 <= relative_risk_source['relative_risk'] <= 100:
            raise ValueError(f"If specifying a single value for relative risk, it should be in the "
                             f"range [1, 100]. You provided {relative_risk_source['relative_risk']} for {source_key}.")
    elif source_type == 'normal distribution':
        if relative_risk_source['mean'] <= 0 or relative_risk_source['se'] <= 0:
            raise ValueError(f"To specify parameters for a normal distribution for a risk effect, you must provide"
                             f"both mean and se above 0. This is not the case for {source_key}.")
    elif source_type == 'log distribution':
        if relative_risk_source['log_mean'] <= 0 or relative_risk_source['log_se'] <= 0:
            raise ValueError(f"To specify parameters for a log distribution for a risk effect, you must provide"
                             f"both log_mean and log_se above 0. This is not the case for {source_key}.")
        if relative_risk_source['tau_squared'] < 0:
            raise ValueError(f"To specify parameters for a log distribution for a risk effect, you must provide"
                             f"tau_squared >= 0. This is not the case for {source_key}.")
    else:
        pass

    return source_type

