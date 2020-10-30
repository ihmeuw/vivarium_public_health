"""
=========================
Risk Data Transformations
=========================

This module contains tools for handling raw risk exposure and relative
risk data and performing any necessary data transformations.

"""
from typing import Union

import numpy as np
import pandas as pd

from vivarium_public_health.utilities import EntityString, TargetString


#############
# Utilities #
#############

def pivot_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """Pivots data that is long on categories to be wide."""
    key_cols = ['sex', 'age_start', 'age_end', 'year_start', 'year_end']
    key_cols = [k for k in key_cols if k in data.columns]
    data = data.pivot_table(index=key_cols, columns='parameter', values='value').reset_index()
    data.columns.name = None
    return data


##########################
# Exposure data handlers #
##########################

def get_distribution_data(builder, risk: EntityString):
    validate_distribution_data_source(builder, risk)
    data = load_distribution_data(builder, risk)
    return data


def get_exposure_post_processor(builder, risk: EntityString):
    thresholds = builder.configuration[risk.name]['category_thresholds']

    if thresholds:
        thresholds = [-np.inf] + thresholds + [np.inf]
        categories = [f'cat{i}' for i in range(1, len(thresholds))]

        def post_processor(exposure, _):
            return pd.Series(pd.cut(exposure, thresholds, labels=categories),
                             index=exposure.index).astype(str)
    else:
        post_processor = None

    return post_processor


def load_distribution_data(builder, risk: EntityString):
    exposure_data = get_exposure_data(builder, risk)

    data = {'distribution_type': get_distribution_type(builder, risk),
            'exposure': exposure_data,
            'exposure_standard_deviation': get_exposure_standard_deviation_data(builder, risk),
            'weights': get_exposure_distribution_weights(builder, risk)}
    return data


def get_distribution_type(builder, risk: EntityString):
    risk_config = builder.configuration[risk.name]

    if risk_config['exposure'] == 'data' and not risk_config['rebinned_exposed']:
        distribution_type = builder.data.load(f'{risk}.distribution')
    else:
        distribution_type = 'dichotomous'

    return distribution_type


def get_exposure_data(builder, risk: EntityString):
    exposure_data = load_exposure_data(builder, risk)
    exposure_data = rebin_exposure_data(builder, risk, exposure_data)

    if get_distribution_type(builder, risk) in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        exposure_data = pivot_categorical(exposure_data)

    return exposure_data


def load_exposure_data(builder, risk: EntityString):
    risk_config = builder.configuration[risk.name]
    exposure_source = risk_config['exposure']

    if exposure_source == 'data':
        exposure_data = builder.data.load(f'{risk}.exposure')
    else:
        if isinstance(exposure_source, str):  # Build from covariate
            cat1 = builder.data.load(f'{exposure_source}.estimate')
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

    return exposure_data


def get_exposure_standard_deviation_data(builder, risk: EntityString):
    distribution_type = get_distribution_type(builder, risk)
    if distribution_type in ['normal', 'lognormal', 'ensemble']:
        exposure_sd = builder.data.load(f'{risk}.exposure_standard_deviation')
    else:
        exposure_sd = None
    return exposure_sd


def get_exposure_distribution_weights(builder, risk: EntityString):
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


def rebin_exposure_data(builder, risk: EntityString, exposure_data: pd.DataFrame):
    validate_rebin_source(builder, risk, exposure_data)
    rebin_exposed_categories = set(builder.configuration[risk.name]['rebinned_exposed'])

    if rebin_exposed_categories:
        exposure_data = _rebin_exposure_data(exposure_data, rebin_exposed_categories)

    return exposure_data


def _rebin_exposure_data(exposure_data: pd.DataFrame, rebin_exposed_categories: set) -> pd.DataFrame:
    exposure_data["parameter"] = (exposure_data["parameter"]
                                  .map(lambda p: 'cat1' if p in rebin_exposed_categories else 'cat2'))
    return exposure_data.groupby(list(exposure_data.columns.difference(['value']))).sum().reset_index()


###############################
# Relative risk data handlers #
###############################

def get_relative_risk_data(builder, risk: EntityString, target: TargetString):
    source_type = validate_relative_risk_data_source(builder, risk, target)
    relative_risk_data = load_relative_risk_data(builder, risk, target, source_type)
    relative_risk_data = rebin_relative_risk_data(builder, risk, relative_risk_data)

    if get_distribution_type(builder, risk) in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        relative_risk_data = pivot_categorical(relative_risk_data)

    else:
        relative_risk_data = relative_risk_data.drop(['parameter'], 'columns')

    return relative_risk_data


def load_relative_risk_data(builder, risk: EntityString, target: TargetString, source_type: str):
    relative_risk_source = builder.configuration[f'effect_of_{risk.name}_on_{target.name}'][target.measure]

    if source_type == 'data':
        relative_risk_data = builder.data.load(f'{risk}.relative_risk')
        correct_target = ((relative_risk_data['affected_entity'] == target.name)
                          & (relative_risk_data['affected_measure'] == target.measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))

    elif source_type == 'relative risk value':
        relative_risk_data = _make_relative_risk_data(builder, float(relative_risk_source['relative_risk']))

    else:  # distribution
        parameters = {k: v for k, v in relative_risk_source.to_dict().items() if v is not None}
        random_state = np.random.RandomState(
            builder.randomness.get_seed(f'effect_of_{risk.name}_on_{target.name}.{target.measure}')
        )
        cat1_value = generate_relative_risk_from_distribution(random_state, parameters)
        relative_risk_data = _make_relative_risk_data(builder, cat1_value)

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
        log_value = parameters['log_mean'] + parameters['log_se']*random_state.randn()
        if parameters['tau_squared']:
            log_value += random_state.normal(0, parameters['tau_squared'])
        rr_value = np.exp(log_value)
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


def rebin_relative_risk_data(builder, risk: EntityString, relative_risk_data: pd.DataFrame) -> pd.DataFrame:
    """ When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
        After rebinning, rr for both exposed and unexposed categories should be the weighted sum of relative risk
        of the component categories where weights are relative proportions of exposure of those categories.
        For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
        for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
        (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
    """
    rebin_exposed_categories = set(builder.configuration[risk.name]['rebinned_exposed'])
    validate_rebin_source(builder, risk, relative_risk_data)

    if rebin_exposed_categories:
        exposure_data = load_exposure_data(builder, risk)
        relative_risk_data = _rebin_relative_risk_data(relative_risk_data, exposure_data, rebin_exposed_categories)

    return relative_risk_data


def _rebin_relative_risk_data(relative_risk_data: pd.DataFrame, exposure_data: pd.DataFrame,
                              rebin_exposed_categories: set) -> pd.DataFrame:
    cols = list(exposure_data.columns.difference(['value']))

    relative_risk_data = relative_risk_data.merge(exposure_data, on=cols)
    relative_risk_data['value_x'] = relative_risk_data.value_x.multiply(relative_risk_data.value_y)
    relative_risk_data.parameter = (relative_risk_data["parameter"]
                                    .map(lambda p: 'cat1' if p in rebin_exposed_categories else 'cat2'))
    relative_risk_data = relative_risk_data.groupby(cols).sum().reset_index()
    relative_risk_data['value'] = relative_risk_data.value_x.divide(relative_risk_data.value_y).fillna(0)
    return relative_risk_data.drop(['value_x', 'value_y'], 'columns')


def get_exposure_effect(builder, risk: EntityString):
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

def get_population_attributable_fraction_data(builder, risk: EntityString, target: TargetString):
    exposure_source = builder.configuration[f'{risk.name}']['exposure']
    rr_source_type = validate_relative_risk_data_source(builder, risk, target)

    if exposure_source == 'data' and rr_source_type == 'data' and risk.type == 'risk_factor':
        paf_data = builder.data.load(f'{risk}.population_attributable_fraction')
        correct_target = ((paf_data['affected_entity'] == target.name)
                          & (paf_data['affected_measure'] == target.measure))
        paf_data = (paf_data[correct_target]
                    .drop(['affected_entity', 'affected_measure'], 'columns'))
    else:
        key_cols = ['sex', 'age_start', 'age_end', 'year_start', 'year_end']
        exposure_data = get_exposure_data(builder, risk).set_index(key_cols)
        relative_risk_data = get_relative_risk_data(builder, risk, target).set_index(key_cols)
        mean_rr = (exposure_data * relative_risk_data).sum(axis=1)
        paf_data = ((mean_rr - 1)/mean_rr).reset_index().rename(columns={0: 'value'})
    return paf_data


##############
# Validators #
##############

def validate_distribution_data_source(builder, risk: EntityString):
    """Checks that the exposure distribution specification is valid."""
    exposure_type = builder.configuration[risk.name]['exposure']
    rebin = builder.configuration[risk.name]['rebinned_exposed']
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


def validate_relative_risk_data_source(builder, risk: EntityString, target: TargetString):
    source_key = f'effect_of_{risk.name}_on_{target.name}'
    relative_risk_source = builder.configuration[source_key][target.measure]

    provided_keys = set(k for k, v in relative_risk_source.to_dict().items() if isinstance(v, (int, float)))

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


def validate_rebin_source(builder, risk: EntityString, data: pd.DataFrame):
    rebin_exposed_categories = set(builder.configuration[risk.name]['rebinned_exposed'])

    if rebin_exposed_categories and builder.configuration[risk.name]['category_thresholds']:
        raise ValueError(f'Rebinning and category thresholds are mutually exclusive. '
                         f'You provided both for {risk.name}.')

    if rebin_exposed_categories and 'polytomous' not in builder.data.load(f'{risk}.distribution'):
        raise ValueError(f'Rebinning is only supported for polytomous risks. You provided rebinning exposed categories'
                         f'for {risk.name}, which is of type {builder.data.load(f"{risk}.distribution")}.')

    invalid_cats = rebin_exposed_categories.difference(set(data.parameter))
    if invalid_cats:
        raise ValueError(f'The following provided categories for the rebinned exposed category of {risk.name} '
                         f'are not found in the exposure data: {invalid_cats}.')

    if rebin_exposed_categories == set(data.parameter):
        raise ValueError(f'The provided categories for the rebinned exposed category of {risk.name} comprise all '
                         f'categories for the exposure data. At least one category must be left out of the provided '
                         f'categories to be rebinned into the unexposed category.')
