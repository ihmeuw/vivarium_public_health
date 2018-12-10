import numpy as np
import pandas as pd

from vivarium.config_tree import ConfigTree
from vivarium_inputs import core
from vivarium_inputs.utilities import normalize_for_simulation, get_age_group_bins_from_age_group_id
import itertools


def should_rebin(risk: str, config: ConfigTree) -> bool:
    """ Check the configuration whether to rebin the polytomous risk """

    return (risk in config) and ('rebin' in config[risk]) and (config[risk].rebin)


def rebin_exposure_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Rebin the polytomous risk and have only cat1/cat2 exposure data """

    unexposed = sorted([c for c in data['parameter'].unique() if 'cat' in c], key=lambda c: int(c[3:]))[-1]
    middle_cats = set(data['parameter']) - {unexposed} - {'cat1'}
    data['sex'] = data['sex'].astype(object)
    df = data.groupby(['year_start', 'year_end', 'age_group_start', 'age_group_end', 'sex'], as_index=False)
    assert np.allclose(df['value'].sum().value, 1)

    def rebin(g):
        g.reset_index(inplace=True)
        to_drop = g['parameter'].isin(middle_cats)
        g.drop(g[to_drop].index, inplace=True)

        g.loc[g.parameter == 'cat1', 'value'] = 1 - g[g.parameter == unexposed].loc[:, 'value'].values
        return g

    df = df.apply(rebin).reset_index()
    return df.replace(unexposed, 'cat2')


def rebin_rr_data(rr: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """ When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
        For the exposed categories of relative risk (after rebinning) should be the weighted sum of relative risk
        of those categories where weights are relative proportions of exposure of those categories.

        For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
        for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
        (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
    """

    df = exposure.merge(rr, on=['parameter', 'sex', 'age_group_start', 'age_group_end',
                                'year_start', 'year_end'])

    df = df.groupby(['year_start', 'year_end', 'age_group_start', 'age_group_end', 'sex'], as_index=False)

    unexposed = sorted([c for c in rr['parameter'].unique() if 'cat' in c], key=lambda c: int(c[3:]))[-1]
    middle_cats = set(rr['parameter']) - {unexposed} - {'cat1'}
    rr['sex'] = rr['sex'].astype(object)
    exposure['sex'] = exposure['sex'].astype(object)

    def rebin_rr(g):
        # value_x = exposure, value_y = rr
        g['weighted_rr'] = g['value_x']*g['value_y']
        x = g['weighted_rr'].sum()-g.loc[g.parameter == unexposed, 'weighted_rr']
        x /= g['value_x'].sum()-g.loc[g.parameter == unexposed, 'value_x'].values
        to_drop = g['parameter'].isin(middle_cats)
        g.drop(g[to_drop].index, inplace=True)
        g.drop(['value_x', 'value_y', 'weighted_rr'], axis=1, inplace=True)
        g['value'] = x.iloc[0]
        g.loc[g.parameter == unexposed, 'value'] = 1.0
        g['value'].fillna(0, inplace=True)
        return g

    df = df.apply(rebin_rr).reset_index().loc[:, ['parameter', 'sex', 'value', 'age_group_start',
                                                  'age_group_end', 'year_start', 'year_end']]
    return df.replace(unexposed, 'cat2')


def get_paf_data(ex: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:

    years = rr.year_start.unique()
    ex = ex[ex['year_start'].isin(years)]
    key_cols = ['sex', 'parameter', 'year_start', 'year_end', 'age_group_start', 'age_group_end']
    df = ex.merge(rr, on=key_cols)
    df = df.groupby(['age_group_start', 'age_group_end', 'sex', 'year_start', 'year_end'], as_index=False)

    def compute_paf(g):
        to_drop = g['parameter'] != 'cat1'
        tmp = g['value_x'] * g['value_y']
        tmp = tmp.sum()
        g.drop(g[to_drop].index, inplace=True)
        g.drop(['parameter', 'value_x', 'value_y'], axis=1, inplace=True)
        g['value'] = (tmp-1)/tmp
        return g

    paf = df.apply(compute_paf).reset_index()
    paf = paf.replace(-np.inf, 0)  # Rows with zero exposure.

    return paf


def exposure_from_covariate(config_name: str, builder) -> pd.DataFrame:
    """For use with DummyRisk component. config_name is the covariate name (or
    1 - covariate name) specified in configuration to use for exposure.
    """
    cn = config_name.split('-')
    if cn[0].rstrip() == '1':
        cov = cn[1].lstrip()
    else:
        cov = config_name

    data = builder.data.load(f'covariate.{cov}.estimate')
    data = data.drop(['lower_value', 'upper_value'], axis='columns')
    data = data.rename(columns={'mean_value': 'value'})

    if cn[0].rstrip() == '1':
        data['value'] = data.value.apply(lambda x: 1-x)

    data['parameter'] = 'cat1'

    cat2 = data.copy()
    cat2['parameter'] = 'cat2'
    cat2['value'] = cat2.value.apply(lambda x: 1-x)

    return data.append(cat2)


def exposure_rr_from_config_value(value, year_start, year_end, measure) -> pd.DataFrame:
    years = range(year_start, year_end+1)
    age_group_ids = core.get_age_bins().age_group_id
    sexes = ['Male', 'Female']

    list_of_lists = [years, age_group_ids, sexes]
    data = pd.DataFrame(list(itertools.product(*list_of_lists)), columns=['year_id', 'age_group_id', 'sex'])

    data = get_age_group_bins_from_age_group_id(normalize_for_simulation(data))

    cat1 = data.copy()
    cat1['parameter'] = 'cat1'
    cat1['value'] = value

    cat2 = data.copy()
    cat2['parameter'] = 'cat2'
    cat2['value'] = 1 - value if measure == 'exposure' else 1

    return cat1.append(cat2)


def build_exp_data_from_config(builder, risk):
    exp_value = builder.configuration[risk]['exposure']

    if isinstance(exp_value, str):
        exp_data = exposure_from_covariate(exp_value, builder)
    elif isinstance(exp_value, (int, float)):
        if exp_value < 0 or exp_value > 1:
            raise ValueError(f"The specified value for {risk} exposure should be in the range [0, 1]. "
                             f"You specified {exp_value}")

        exp_data = exposure_rr_from_config_value(exp_value, builder.configuration.time.start.year,
                                                 builder.configuration.time.end.year, 'exposure')
    else:
        raise TypeError(f"You may only specify a value for {risk} exposure that is the "
                        f"name of a covariate or a single value. You specified {exp_value}.")
    return exp_data
