import numpy as np
import pandas as pd
import pytest

from vivarium_public_health.risks import distributions

@pytest.fixture
def test_risk_factor(mocker):

    test_rf = mocker.MagicMock()
    test_rf.test_risk = dict()
    test_rf.test_risk['distribution'] = 'ensemble'
    exposure_mean = pd.DataFrame({'year': [1990]*4, 'sex': [1]*4,
                                  'age': [2, 3, 4, 5], 'value': [10, 20, 30, 40]})
    exposure_sd = pd.DataFrame({'year': [1990]*4, 'sex': [1]*4,
                                'age': [2, 3, 4, 5], 'value': [1, 3, 5, 7]})
    weights = {'betasr': 0.1, 'exp': 0.1, 'gamma': 0.1, 'gumbel': 0.1, 'invgamma': 0.1,
               'llogis': 0.1,  'lnorm': 0.1, 'mgamma': 0.1,  'mgumbel': 0.1,
               'norm': 0.03, 'invweibull': 0.03, 'weibull': 0.04}
    ensemble_weights = pd.DataFrame(weights, index=[0])
    test_rf.test_risk['exposure']= exposure_mean
    test_rf.test_risk['exposure_standard_deviation'] = exposure_sd
    test_rf.test_risk['ensemble_weights'] = ensemble_weights
    return test_rf.test_risk

def test_get_min_max():
    test_exposure = pd.DataFrame({'mean': [5, 10, 20, 50, 100], 'standard_deviation': [1, 3, 5, 10, 15]}, index=range(5))
    expected = dict()
    expected['x_min'] = np.array([2.6586837, 3.86641019, 9.06608812, 26.58683698, 62.37010755])
    expected['x_max'] = np.array([9.0414898, 23.72824267, 41.5251411, 90.41489799, 156.80510239])
    test = distributions.BaseDistribution._get_min_max(test_exposure)

    assert np.allclose(test['x_min'], expected['x_min'])
    assert np.allclose(test['x_max'], expected['x_max'])


def test_get_distribution(test_risk_factor, mocker):
    builder = mocker.MagicMock()
    builder.data.load.side_effect = lambda args: test_risk_factor[args.split('.')[-1]]

    # just keep the start weight that we provide to make a test ensemble object
    initial_weight = test_risk_factor['ensemble_weights'].iloc[0]
    ensemble = distributions.get_distribution('test_risk', 'test_risk_factor', builder)

    # check whether we start with correct weight
    assert np.isclose(np.sum(initial_weight), 1)

    # then check whether it properly drops 'invweibull' and rescale it
    assert np.isclose(np.sum(ensemble.weights), 1)
    assert 'invweibull' not in ensemble.weights
    assert 'invweibull' not in ensemble._distributions

    expected_weight = initial_weight.drop('invweibull')
    expected_weight = expected_weight/np.sum(expected_weight)

    for key in expected_weight.keys():
        assert np.isclose(expected_weight[key], ensemble.weights[key])


@pytest.mark.parametrize('exposure_idx', [0, 1, 2, 3])
def test_individual_distribution(exposure_idx):
    exposure_level = [(10, 1), (20, 3), (30, 5), (40, 7)]
    expected = dict()
    generated = dict()
    # now look into the details of each distribution parameters
    # this is a dictionary of distributions considered for ensemble distribution
    m, s = exposure_level[exposure_idx]
    e = pd.DataFrame({'mean': m, 'standard_deviation': s}, index=[0])

    # Beta
    beta = distributions.Beta(e)

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
            assert np.isclose(expected[dist][params][exposure_idx], generated[dist][params])
