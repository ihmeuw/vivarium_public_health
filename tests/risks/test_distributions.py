import numpy as np
import pandas as pd
import pytest

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.testing_utilities import build_table
from vivarium_public_health.risks import distributions
from vivarium_public_health.util import pivot_age_sex_year_binned


def test_get_min_max():
    test_exposure = pd.DataFrame({'mean': [5, 10, 20, 50, 100], 'standard_deviation': [1, 3, 5, 10, 15]}, index=range(5))
    expected = dict()
    expected['x_min'] = np.array([2.6586837, 3.86641019, 9.06608812, 26.58683698, 62.37010755])
    expected['x_max'] = np.array([9.0414898, 23.72824267, 41.5251411, 90.41489799, 156.80510239])
    test = distributions.BaseDistribution._get_min_max(test_exposure)

    assert np.allclose(test['x_min'], expected['x_min'])
    assert np.allclose(test['x_max'], expected['x_max'])


def test_get_distribution_ensemble_risk(mocker):

    ensemble_mean = []
    ensemble_sd = []
    for m,s in zip([10, 20, 30, 40], [1, 3, 5, 7]):
        ensemble_mean.append(build_table(m, 1990, 1992, ('age','year', 'sex', 'parameter', 'value')))
        ensemble_sd.append(build_table(s, 1990, 1992, ('age', 'year', 'sex', 'parameter', 'value')))

    ensemble_weights = {'betasr': 0.1, 'exp': 0.1, 'gamma': 0.1, 'gumbel': 0.1, 'invgamma': 0.1, 'llogis': 0.1,
                        'lnorm': 0.1, 'mgamma': 0.1,  'mgumbel': 0.1, 'norm': 0.03, 'invweibull': 0.03, 'weibull': 0.04}

    cols = ('age', 'year', 'sex')
    weights = []
    for k, v in ensemble_weights.items():
        cols += (k,)
        weights.append(v)

    ensemble_mean = pd.concat(ensemble_mean)
    ensemble_sd = pd.concat(ensemble_sd)
    ensemble_w = build_table(weights, 1990, 1992, cols)

    mock_data = {'distribution': 'ensemble', 'exposure': ensemble_mean, 'exposure_standard_deviation': ensemble_sd,
                 'ensemble_weights': ensemble_w}

    builder = mocker.MagicMock()
    builder.data.load.side_effect = lambda _: mock_data.get(_.split('.')[-1])

    e = distributions.get_distribution('risk_factor', 'test_risk', builder)

    # check whether we start with correct weight
    assert np.isclose(np.sum(e.weights.sum()), 1)

    assert 'invweibull' not in e.weights
    assert 'invweibull' not in e._distributions

    ensemble_weights.pop('invweibull')

    for k,v in ensemble_weights.items():
        ensemble_weights[k] = v/sum(ensemble_weights.values())
        np.isclose(ensemble_weights[k], e.weights[k])


# NOTE: This test is to ensure that our math to find the parameters for each distribution is correct.
exposure_levels = [(0, 10, 1), (1, 20, 3), (2, 30, 5), (3, 40, 7)]
@pytest.mark.parametrize('i, mean, sd', exposure_levels)
def test_individual_distribution(i, mean, sd):
    expected = dict()
    generated = dict()
    # now look into the details of each distribution parameters
    # this is a dictionary of distributions considered for ensemble distribution
    e = pd.DataFrame({'mean': mean, 'standard_deviation': sd}, index=[0])

    # Beta
    beta = d.Beta(e)

    generated['betasr'] = beta._parameter_data
    expected['betasr'] = dict()
    expected['betasr']['scale'] = [6.232114, 18.886999, 31.610845, 44.354704]
    expected['betasr']['a'] = [3.679690, 3.387153, 3.291559, 3.244209]
    expected['betasr']['b'] = [4.8479, 5.113158, 5.197285, 5.238462]

    # Exponential
    exp = distributions.Exponential(e)
    generated['exp'] = exp._parameter_data

    expected['exp'] = dict()
    expected['exp']['scale'] = [10, 20, 30, 40]


    # Gamma
    gamma = distributions.Gamma(e)
    generated['gamma'] = gamma._parameter_data

    expected['gamma'] = dict()
    expected['gamma']['a'] =[100, 44.444444, 36, 32.653061]
    expected['gamma']['scale'] = [0.1, 0.45, 0.833333, 1.225]

    # Gumbel
    gumbel = distributions.Gumbel(e)
    generated['gumbel'] = gumbel._parameter_data

    expected['gumbel'] = dict()
    expected['gumbel']['loc'] =[9.549947, 18.649840, 27.749734, 36.849628]
    expected['gumbel']['scale'] = [0.779697, 2.339090, 3.898484, 5.457878]

    # InverseGamma
    invgamma = distributions.InverseGamma(e)
    generated['invgamma'] = invgamma._parameter_data

    expected['invgamma'] = dict()
    expected['invgamma']['a'] = [102.000001, 46.444443, 38.000001, 34.653062]
    expected['invgamma']['scale'] = [1010.000013, 908.888853, 1110.000032, 1346.122489]

    # LogLogistic
    llogis = distributions.LogLogistic(e)
    generated['llogis'] = llogis._parameter_data

    expected['llogis'] = dict()
    expected['llogis']['c'] = [18.246506, 12.254228, 11.062771, 10.553378]
    expected['llogis']['d'] = [1, 1, 1, 1]
    expected['llogis']['scale'] = [9.950669, 19.781677, 29.598399, 39.411819]

    # LogNormal
    lnorm = distributions.LogNormal(e)
    generated['lnorm'] = lnorm._parameter_data

    expected['lnorm'] = dict()
    expected['lnorm']['s'] = [0.099751, 0.149166, 0.165526, 0.173682]
    expected['lnorm']['scale'] = [9.950372, 19.778727, 29.591818, 39.401219]

    # MirroredGumbel
    mgumbel = distributions.MirroredGumbel(e)
    generated['mgumbel'] = mgumbel._parameter_data

    expected['mgumbel'] = dict()
    expected['mgumbel']['loc'] = [3.092878, 10.010861, 17.103436, 24.240816]
    expected['mgumbel']['scale'] = [0.779697,2.339090,3.898484, 5.457878]

    # MirroredGamma
    mgamma = distributions.MirroredGamma(e)
    generated['mgamma'] = mgamma._parameter_data

    expected['mgamma'] = dict()
    expected['mgamma']['a'] = [12.552364, 14.341421, 14.982632, 15.311779]
    expected['mgamma']['scale'] = [0.282252, 0.792182, 1.291743, 1.788896]


    # Normal
    norm = distributions.Normal(e)
    generated['norm'] = norm._parameter_data

    expected['norm'] = dict()
    expected['norm']['loc'] = [10, 20, 30, 40]
    expected['norm']['scale'] = [1, 3, 5, 7]

    # Weibull
    weibull = distributions.Weibull(e)
    generated['weibull'] = weibull._parameter_data

    expected['weibull'] = dict()
    expected['weibull']['c'] = [12.153402, 7.906937, 7.061309, 6.699559]
    expected['weibull']['scale'] = [10.430378, 21.249309, 32.056036, 42.859356]

    for dist in expected.keys():
        for params in expected[dist].keys():
            assert np.isclose(expected[dist][params][i], generated[dist][params])


def test_should_rebin():
    test_config = build_simulation_configuration()
    test_config['population'] = {'population_size': 100}
    assert not distributions.should_rebin('test_risk', test_config)

    test_config['test_risk'] = {}
    assert not distributions.should_rebin('test_risk', test_config)

    test_config['test_risk'].rebin = False
    assert not distributions.should_rebin('test_risk', test_config)

    test_config['test_risk']['rebin'] = True
    assert distributions.should_rebin('test_risk', test_config)


def test_rebin_exposure():
    cats = ['cat1', 'cat2', 'cat3', 'cat4']
    year_start = 2010
    year_end = 2013

    wrong_values = [0.1, 0.1, 0.1, 0.1]

    wrong_df = []
    for cat, value in zip(cats, wrong_values):
        wrong_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
    wrong_df = pd.concat(wrong_df)

    with pytest.raises(AssertionError):
        distributions.rebin_exposure_data(wrong_df)

    values = [0.1, 0.2, 0.3, 0.4]
    test_df = []
    for cat, value in zip(cats, values):
        test_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
    test_df = pd.concat(test_df)

    expected = []

    for cat, value in zip (['cat1', 'cat2'], [0.6, 0.4]):
        expected.append(build_table([cat, value], year_start, year_end, ('age', 'year', 'sex', 'parameter', 'value')))

    expected = pd.concat(expected).loc[:, ['age', 'year', 'sex', 'parameter', 'value']]
    rebinned = d.rebin_exposure_data(test_df).loc[:, expected.columns]
    expected = expected.set_index(['age', 'year','sex'])
    rebinned = rebinned.set_index(['age', 'year', 'sex'])

    assert np.allclose(expected.value[expected.parameter == 'cat1'], rebinned.value[rebinned.parameter=='cat1'])
    assert np.allclose(expected.value[expected.parameter == 'cat2'], rebinned.value[rebinned.parameter == 'cat2'])


def test_get_distribution_dichotomous_risk(mocker):

    test_exposure = []
    for cat, value in zip(['cat1', 'cat2'], [0.2, 0.8]):
        test_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))

    test_exposure = pd.concat(test_exposure)
    mock_data = {'exposure': test_exposure, 'distribution': 'dichotomous'}

    builder = mocker.MagicMock()
    builder.data.load.side_effect = lambda _: mock_data.get(_.split('.')[-1])

    test_d = distributions.get_distribution('dichotomous_risk', 'risk_factor', builder)
    Dichotomous_d = distributions.DichotomousDistribution(pivot_age_sex_year_binned(test_exposure, 'parameter', 'value'),
                                                          'dichotomous_risk')

    assert type(test_d) == type(Dichotomous_d)
    assert test_d._risk == Dichotomous_d._risk
    assert test_d.exposure_data.equals(Dichotomous_d.exposure_data)


def test_get_distribution_polytomous_risk(mocker):

    test_exposure = []
    for cat, value in zip(['cat1', 'cat2', 'cat3', 'cat4'], [0.2, 0.3, 0.1, 0.4]):
        test_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))

    test_exposure = pd.concat(test_exposure)
    mock_data = {'exposure': test_exposure, 'distribution': 'polytomous'}

    builder = mocker.MagicMock()
    builder.data.load.side_effect = lambda _: mock_data.get(_.split('.')[-1])

    test_d = distributions.get_distribution('polytomous_risk', 'risk_factor', builder)
    Polytomous_d = distributions.PolytomousDistribution(pivot_age_sex_year_binned(test_exposure, 'parameter', 'value'),
                                                        'polytomous_risk')

    assert type(test_d) == type(Polytomous_d)
    assert test_d._risk == Polytomous_d._risk
    assert test_d.categories == Polytomous_d.categories
    assert test_d.exposure_data.equals(Polytomous_d.exposure_data)


def test_get_distribution_polytomous_risk_rebinned(mocker):
    rebin_mock = mocker.patch('vivarium_public_health.risks.distributions.should_rebin')
    rebin_mock.return_value = True
    test_exposure = []
    for cat, value in zip(['cat1', 'cat2', 'cat3', 'cat4'], [0.2, 0.3, 0.1, 0.4]):
        test_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))

    test_exposure = pd.concat(test_exposure)
    mock_data = {'exposure': test_exposure, 'distribution': 'polytomous'}

    builder = mocker.MagicMock()
    builder.data.load.side_effect = lambda _: mock_data.get(_.split('.')[-1])

    test_d = distributions.get_distribution('polytomous_risk', 'risk_factor', builder)

    rebinned_exposure = []
    for cat, value in zip(['cat1', 'cat2'], [0.6, 0.4]):
        rebinned_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))
    rebinned_exposure = pd.concat(rebinned_exposure)

    Polytomous_d = distributions.RebinPolytomousDistribution(pivot_age_sex_year_binned(rebinned_exposure, 'parameter',
                                                                                       'value'),'polytomous_risk')

    assert type(test_d) == type(Polytomous_d)
    assert test_d._risk == Polytomous_d._risk
    assert test_d.categories == Polytomous_d.categories
    assert test_d.exposure_data.equals(Polytomous_d.exposure_data)
