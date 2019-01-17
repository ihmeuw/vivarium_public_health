import numpy as np
import pandas as pd

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
        weights = weights[['parameter', 'value']].drop_duplicates().set_index('parameter').transpose()
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

def get_relative_risk_data(builder, risk: RiskString, target: TargetString):
    validate_relative_risk_data_source(builder, risk, target)
    relative_risk_data = load_relative_risk_data(builder, risk, target)
    relative_risk_data = rebin_relative_risk_data(builder, risk, relative_risk_data)
    return relative_risk_data


def load_relative_risk_data(builder, risk: RiskString, target: TargetString):
    distribution_type = get_distribution_type(builder, risk)
    relative_risk_source = builder.configuration[f'effect_of_{risk.name}_on_{target.name}'][target.measure]

    if relative_risk_source == 'data':
        relative_risk_data = builder.data.load(f'{risk}.relative_risk')
        correct_target = ((relative_risk_data['affected_entity'] == target.name)
                          & (relative_risk_data['affected_measure'] == target.measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))
    else:
        cat1 = builder.data.load('population.demographic_dimensions')
        cat1['parameter'] = 'cat1'
        cat1['value'] = float(relative_risk_source)
        cat2 = cat1.copy()
        cat2['parameter'] = 'cat2'
        cat2['value'] = 1 - cat2['value']
        relative_risk_data = pd.concat([cat1, cat2], ignore_index=True)

    if distribution_type in ['dichotomous', 'ordered polytomous', 'unordered polytomous']:
        relative_risk_data = pivot_categorical(relative_risk_data)

    return relative_risk_data


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
        raise NotImplementedError()
        # tmred = builder.data.load(f"{risk_type}.{risk}.tmred")
        # tmrel = 0.5 * (tmred["min"] + tmred["max"])
        # exposure_parameters = builder.data.load(f"{risk_type}.{risk}.exposure_parameters")
        # max_exposure = exposure_parameters["max_rr"]
        # scale = exposure_parameters["scale"]
        #
        # def exposure_effect(rates, rr):
        #     exposure = np.minimum(risk_exposure(rr.index), max_exposure)
        #     relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
        #     return rates * relative_risk
    else:
        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            return rates * (rr.lookup(exposure.index, exposure))

    return exposure_effect


##################################################
# Population attributable fraction data handlers #
##################################################

def get_population_attributable_fraction_data(builder, risk: RiskString, target: TargetString):
    exposure_source = builder.configuration[f'{risk.name}']['exposure']
    rr_source = builder.configuration[f'effect_of_{risk.name}_on_{target.name}'][target.measure]

    if exposure_source == 'data' and rr_source == 'data' and risk.type == 'risk_factor':
        paf_data = builder.data.load(f'{risk}.population_attributable_fraction')
        correct_target = ((paf_data['affected_entity'] == target.name)
                          & (paf_data['affected_measure'] == target.measure))
        paf_data = (paf_data[correct_target]
                    .drop(['affected_entity', 'affected_measure'], 'columns'))
    else:
        key_cols = ['sex', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
        exposure_data = get_exposure_data(builder, risk).set_index(key_cols + ['parameter'])
        relative_risk_data = get_relative_risk_data(builder, risk, target).set_index(key_cols + ['parameter'])
        weighted_rr = (exposure_data * relative_risk_data).reset_index()
        mean_rr = weighted_rr.groupby(key_cols).apply(lambda sub_df: sub_df.value.sum())
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

    elif risk.type in ['risk_factor', 'coverage_gap'] and exposure_type != 'data':
        if isinstance(exposure_type, (int, float)) and not 0 <= exposure_type <= 1:
            raise ValueError(f"Exposure should be in the range [0, 1]")
        elif isinstance(exposure_type, str) and exposure_type.split('.')[0] != 'covariate':
            raise ValueError(f"Exposure must be specified as 'data', an integer or float value, "
                             f"or as a string in the format covariate.covariate_name")
        else:
            pass  # All good
    else:
        raise ValueError(f'Unknown risk type {risk.type} for risk {risk.name}')


def validate_relative_risk_data_source(builder, risk: RiskString, target: TargetString):
    relative_risk_source = builder.configuration[f'effect_of_{risk.name}_on_{target.name}'][target.measure]

    if isinstance(relative_risk_source, (int, float)) and not 1 <= relative_risk_source <= 100:
        raise ValueError(f"Relative risk should be in the range [1, 100]")
    elif relative_risk_source == 'data':
        pass
    else:
        raise ValueError(f'Invalid risk effect specification for risk {risk.name} and target {target.name}')
