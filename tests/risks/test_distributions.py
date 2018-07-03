import numpy as np
import pandas as pd
import pytest

from ceam_public_health.risks import distributions

@pytest.fixture
def test_risk_factor(mocker):
    test_rf = mocker.MagicMock()
    test_rf.test_risk = dict()
    test_rf.test_risk['distribution'] = 'ensemble'
    exposure_mean = pd.DataFrame({'year':[1990]*4, 'sex': [1]*4,
                                  'age':[2, 3, 4, 5], 'value':[10, 20, 30, 40]})
    exposure_sd = pd.DataFrame({'year':[1990]*4, 'sex': [1]*4,
                                  'age':[2,3,4, 5], 'value':[1, 3, 5, 7]})
    weights = {'betasr': 0.1, 'exp': 0.1, 'gamma': 0.1, 'gumbel': 0.1, 'invgamma': 0.1,
               'llogis':0.1,  'lnorm': 0.1, 'mgamma' : 0.1,  'mgumbel':0.1,
               'norm':0.03, 'invweibull': 0.03, 'weibull': 0.04}
    ensemble_weights = pd.DataFrame(weights, index=exposure_mean.index)
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

    # now look into the details of each distribution parameters
    # this is a dictionary of distributions considered for ensemble distribution
    test_distributions = ensemble._distributions
    expected = {key: dict() for key in test_distributions.keys()}
    generated ={key: dict() for key in test_distributions.keys()}

    # Beta
    expected['betasr']['scale'] = pd.DataFrame([6.232114, 18.886999, 31.610845, 44.354704], index=range(4))
    expected['betasr']['a'] = pd.DataFrame([3.679690, 3.387153, 3.291559, 3.244209], index=range(4))
    expected['betasr']['b'] = pd.DataFrame([4.8479, 5.113158, 5.197285, 5.238462], index=range(4))
    generated['betasr'] = test_distributions['betasr']._parameter_data

    # Exponential
    expected['exp']['scale'] = pd.DataFrame([10, 20, 30, 40], index=range(4))
    generated['exp'] = test_distributions['exp']._parameter_data

    # Gamma
    expected['gamma']['a'] = pd.DataFrame([100, 44.444444, 36, 32.653061], index=range(4))
    expected['gamma']['scale'] = pd.DataFrame([0.1, 0.45, 0.833333, 1.225], index=range(4))
    generated['gamma'] = test_distributions['gamma']._parameter_data

    # Gumbel
    expected['gumbel']['loc'] = pd.DataFrame([9.549947, 18.649840, 27.749734, 36.849628], index=range(4))
    expected['gumbel']['scale'] = pd.DataFrame([0.779697, 2.339090, 3.898484, 5.457878], index=range(4))
    generated['gumbel'] = test_distributions['gumbel']._parameter_data

    # InverseGamma
    expected['invgamma']['a'] = pd.DataFrame([102.000001, 46.444443, 38.000001, 34.653062], index=range(4))
    expected['invgamma']['scale'] = pd.DataFrame([1010.000013, 908.888853, 1110.000032, 1346.122489], index=range(4))
    generated['invgamma'] = test_distributions['invgamma']._parameter_data

    # LogLogistic
    expected['llogis']['c'] = pd.DataFrame([18.246506, 12.254228, 11.062771, 10.553378], index=range(4))
    expected['llogis']['d'] = pd.DataFrame([1, 1, 1, 1], index=range(4))
    expected['llogis']['scale'] = pd.DataFrame([9.950669, 19.781677, 29.598399, 39.411819], index=range(4))
    generated['llogis'] = test_distributions['llogis']._parameter_data

    # LogNormal
    expected['lnorm']['s'] = pd.DataFrame([0.099751, 0.149166, 0.165526, 0.173682], index=range(4))
    expected['lnorm']['scale'] = pd.DataFrame([9.950372, 19.778727, 29.591818, 39.401219], index=range(4))
    generated['lnorm'] = test_distributions['lnorm']._parameter_data

    # MirroredGumbel
    expected['mgumbel']['loc'] = pd.DataFrame([3.092878, 10.010861, 17.103436, 24.240816], index=range(4))
    expected['mgumbel']['scale'] = pd.DataFrame([0.779697,2.339090,3.898484, 5.457878], index=range(4))
    generated['mgumbel'] = test_distributions['mgumbel']._parameter_data

    # MirroredGamma
    expected['mgamma']['a'] = pd.DataFrame([12.552364, 14.341421, 14.982632, 15.311779], index=range(4))
    expected['mgamma']['scale'] = pd.DataFrame([0.282252, 0.792182, 1.291743, 1.788896], index=range(4))
    generated['mgamma'] = test_distributions['mgamma']._parameter_data

    # Normal
    expected['norm']['loc'] = pd.DataFrame([10, 20, 30, 40], index=range(4))
    expected['norm']['scale'] = pd.DataFrame([1, 3, 5, 7], index=range(4))
    generated['norm'] = test_distributions['norm']._parameter_data

    # Weibull
    expected['weibull']['c'] = pd.DataFrame([12.153402, 7.906937, 7.061309, 6.699559], index=range(4))
    expected['weibull']['scale'] = pd.DataFrame([10.430378, 21.249309, 32.056036, 42.859356], index=range(4))
    generated['weibull'] = test_distributions['weibull']._parameter_data

    for dist in expected.keys():
        for params in expected[dist].keys():
            assert np.allclose(expected[dist][params].values, generated[dist][params])

    # test ppf function to generate correct numbers
    test_propensity = np.linspace(0.01, 0.99, 10)
    generated_exposure = ensemble.ppf(test_propensity, interpolation=False)
    expected_exposure = np.array([[6.80974558, 7.95758444, 8.4709367, 8.90777061, 9.33818496,
                                   9.80058963, 10.33772907, 11.02719045, 12.089827, 15.75560794],
                                  [12.62741864, 15.79095391, 17.16011934, 18.29679845, 19.39450681,
                                   20.55367424, 21.87958457, 23.5570375, 26.105012, 34.72951647],
                                  [18.47991571, 23.62724345, 25.84440119, 27.67676612, 29.43937433,
                                   31.29417278, 33.40897947, 36.0763311, 40.11552253, 53.7392625],
                                  [24.34129426, 31.46487207, 34.52807927, 37.05506378, 39.48193045,
                                   42.03202203, 44.93568871, 48.5933309, 54.12509648, 72.75816963]])

    assert np.allclose(generated_exposure, expected_exposure)
