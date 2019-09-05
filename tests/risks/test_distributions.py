# import numpy as np
# import pandas as pd
# import pytest
#
# from vivarium.testing_utilities import build_table
# from vivarium_public_health.risks import distributions
# from vivarium_public_health.risks.data_transformations import pivot_categorical
#
#
# def test_get_distribution_ensemble_risk():
#
#     ensemble_mean = []
#     ensemble_sd = []
#     for m,s in zip([10, 20, 30, 40], [1, 3, 5, 7]):
#         ensemble_mean.append(build_table(m, 1990, 1992, ('age','year', 'sex', 'parameter', 'value')))
#         ensemble_sd.append(build_table(s, 1990, 1992, ('age', 'year', 'sex', 'parameter', 'value')))
#
#     ensemble_weights = {'betasr': 0.1, 'exp': 0.1, 'gamma': 0.1, 'gumbel': 0.1, 'invgamma': 0.1, 'llogis': 0.1,
#                         'lnorm': 0.1, 'mgamma': 0.1,  'mgumbel': 0.1, 'norm': 0.03, 'invweibull': 0.03, 'weibull': 0.04}
#
#     cols = ('age', 'year', 'sex')
#     weights = []
#     for k, v in ensemble_weights.items():
#         cols += (k,)
#         weights.append(v)
#
#     distribution = 'ensemble'
#     ensemble_mean = pd.concat(ensemble_mean)
#     ensemble_sd = pd.concat(ensemble_sd)
#     ensemble_w = build_table(weights, 1990, 1992, cols)
#
#     kwargs = {"exposure_standard_deviation": ensemble_sd, "weights": ensemble_w}
#     e = distributions.get_distribution('risk_factor', distribution, ensemble_mean, **kwargs)
#
#     # check whether we start with correct weight
#     assert np.isclose(np.sum(e.weights.sum()), 1)
#
#     assert 'invweibull' not in e.weights
#     assert 'invweibull' not in e._distributions
#
#     ensemble_weights.pop('invweibull')
#
#     for k,v in ensemble_weights.items():
#         ensemble_weights[k] = v/sum(ensemble_weights.values())
#         np.isclose(ensemble_weights[k], e.weights[k])
#
#
# def test_rebin_exposure():
#     cats = ['cat1', 'cat2', 'cat3', 'cat4']
#     year_start = 2010
#     year_end = 2013
#
#     wrong_values = [0.1, 0.1, 0.1, 0.1]
#
#     wrong_df = []
#     for cat, value in zip(cats, wrong_values):
#         wrong_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
#     wrong_df = pd.concat(wrong_df)
#
#     with pytest.raises(AssertionError):
#         distributions.rebin_exposure_data(wrong_df)
#
#     values = [0.1, 0.2, 0.3, 0.4]
#     test_df = []
#     for cat, value in zip(cats, values):
#         test_df.append(build_table([cat, value], year_start, year_end, ('age','year', 'sex', 'parameter', 'value')))
#     test_df = pd.concat(test_df)
#
#     expected = []
#
#     for cat, value in zip (['cat1', 'cat2'], [0.6, 0.4]):
#         expected.append(build_table([cat, value], year_start, year_end, ('age', 'year', 'sex', 'parameter', 'value')))
#
#     expected = pd.concat(expected).loc[:, ['age_start', 'age_end', 'year_start',
#                                            'year_end', 'sex', 'parameter', 'value']]
#     rebinned = distributions.rebin_exposure_data(test_df).loc[:, expected.columns]
#     expected = expected.set_index(['age_start', 'year_start', 'sex'])
#     rebinned = rebinned.set_index(['age_start', 'year_start', 'sex'])
#
#     assert np.allclose(expected.value[expected.parameter == 'cat1'], rebinned.value[rebinned.parameter=='cat1'])
#     assert np.allclose(expected.value[expected.parameter == 'cat2'], rebinned.value[rebinned.parameter == 'cat2'])
#
#
# def test_get_distribution_dichotomous_risk():
#
#     test_exposure = []
#     for cat, value in zip(['cat1', 'cat2'], [0.2, 0.8]):
#         test_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))
#
#     distribution = 'dichotomous'
#     test_exposure = pd.concat(test_exposure)
#
#     test_d = distributions.get_distribution('dichotomous_risk', distribution, test_exposure)
#     Dichotomous_d = distributions.DichotomousDistribution(pivot_categorical(test_exposure), 'dichotomous_risk')
#
#     assert type(test_d) == type(Dichotomous_d)
#     assert test_d._risk == Dichotomous_d._risk
#     assert test_d.exposure_data.equals(Dichotomous_d.exposure_data)
#
#
# def test_get_distribution_polytomous_risk(mocker):
#     rebin_mock = mocker.patch('vivarium_public_health.risks.distributions.should_rebin')
#     rebin_mock.return_value = False
#
#     test_exposure = []
#     for cat, value in zip(['cat1', 'cat2', 'cat3', 'cat4'], [0.2, 0.3, 0.1, 0.4]):
#         test_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))
#
#
#     distribution = "polytomous"
#     test_exposure = pd.concat(test_exposure)
#     kwargs = {'configuration': None}
#     test_d = distributions.get_distribution('polytomous_risk', distribution, test_exposure, **kwargs)
#     Polytomous_d = distributions.PolytomousDistribution(pivot_categorical(test_exposure), 'polytomous_risk')
#
#     assert type(test_d) == type(Polytomous_d)
#     assert test_d._risk == Polytomous_d._risk
#     assert test_d.categories == Polytomous_d.categories
#     assert test_d.exposure_data.equals(Polytomous_d.exposure_data)
#
#
# def test_get_distribution_polytomous_risk_rebinned(mocker):
#     rebin_mock = mocker.patch('vivarium_public_health.risks.distributions.should_rebin')
#     rebin_mock.return_value = True
#
#     test_exposure = []
#     for cat, value in zip(['cat1', 'cat2', 'cat3', 'cat4'], [0.2, 0.3, 0.1, 0.4]):
#         test_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))
#
#     distribution = "polytomous"
#     test_exposure = pd.concat(test_exposure)
#     kwargs = {"configuration": None}
#     test_d = distributions.get_distribution('polytomous_risk', distribution, test_exposure, **kwargs)
#
#     rebinned_exposure = []
#     for cat, value in zip(['cat1', 'cat2'], [0.6, 0.4]):
#         rebinned_exposure.append(build_table([cat, value], 2000, 2005, ('age', 'year', 'sex', 'parameter', 'value')))
#     rebinned_exposure = pd.concat(rebinned_exposure)
#
#     Polytomous_d = distributions.RebinPolytomousDistribution(pivot_categorical(rebinned_exposure), 'polytomous_risk')
#
#     assert type(test_d) == type(Polytomous_d)
#     assert test_d._risk == Polytomous_d._risk
#     assert test_d.exposure_data.equals(Polytomous_d.exposure_data)
