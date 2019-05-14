import numpy as np
import pandas as pd
from operator import gt, lt
import pytest
from vivarium_public_health.disease import RiskAttributableDisease


@pytest.fixture
def disease_mock(mocker):
    def disease_with_distribution(distribution):
        test_disease = RiskAttributableDisease('cause.test_cause', 'risk_factor.test_risk')
        test_disease.distribution = distribution
        test_disease.population_view = mocker.Mock()
        test_disease._mortality = mocker.Mock()
        return test_disease
    return disease_with_distribution


test_data = [('ordered_polytomous', ['cat1', 'cat2', 'cat3', 'cat4'], ['cat1']),
             ('ordered_polytomous', ['cat1', 'cat2', 'cat3', 'cat4'], ['cat1', 'cat2']),
             ('ordered_polytomous', ['cat1', 'cat2', 'cat3', 'cat4'], ['cat1', 'cat2', 'cat3']),
             ('dichotomous', ['cat1', 'cat2'], ['cat1'])]


@pytest.mark.parametrize('distribution, categories, threshold', test_data)
def test_filter_by_exposure_categorical(disease_mock, mocker, distribution, categories, threshold):
    disease = disease_mock(distribution)
    test_index = range(500)
    per_cat = len(test_index) // len(categories)
    infected = threshold * per_cat
    susceptible = list(set(categories) - set(threshold)) * per_cat
    current_exposure = lambda index: pd.Series(infected + susceptible, index=index)
    filter_func = disease.get_exposure_filter(distribution, current_exposure, threshold)
    expected = lambda index: current_exposure(index).isin(infected)

    assert np.all(expected(test_index) == filter_func(test_index))


test_data = [('ensemble', '>=7'), ('ensemble', '<=7.5'), ('lognormal', '=2.5'), ('normal', '4'), ('normal', '+4'),
             ('lognormal', '>=')]


@pytest.mark.parametrize('distribution, threshold', test_data)
def test_filter_by_exposure_continuous_incorrect_operator(disease_mock, distribution, threshold):
    disease = disease_mock(distribution)
    disease.threshold = threshold
    with pytest.raises(ValueError, match='incorrect threshold'):
        disease.get_exposure_filter(distribution, lambda index: index, threshold)


test_data = [('ensemble', '>7'), ('ensemble', '<5'), ('lognormal', '<3.5'), ('normal', '>5.5')]


@pytest.mark.parametrize('distribution, threshold', test_data)
def test_filter_by_exposure_continuous(disease_mock, distribution, threshold):
    disease = disease_mock(distribution)
    disease.threshold = threshold
    op = {'>', '<'}.intersection(list(threshold)).pop()
    threshold_val = float(threshold.split(op)[-1])
    threshold_op = gt if op == ">" else lt

    test_index = range(500)

    current_exposure = lambda index: pd.Series([threshold_val - 0.2, threshold_val - 0.1, threshold_val,
                                                threshold_val + 0.1, threshold_val + 0.2] *100, index=test_index)

    filter_func = disease.get_exposure_filter(distribution, current_exposure, threshold)
    expected = lambda index: threshold_op(current_exposure(index), threshold_val)

    assert np.all(expected(test_index) == filter_func(test_index))


def test_mortality_rate_pandas_series(disease_mock):
    disease = disease_mock('enesmble')
    num_sims = 500
    test_index = range(num_sims)
    current_disease_status = [disease.cause.name] * int(0.2 * num_sims) + \
                             [f'susceptible_to_{disease.cause.name}'] * int(num_sims * 0.8)
    disease.population_view.get.side_effect =lambda index: pd.DataFrame({disease.cause.name: current_disease_status,
                                                                        'alive': 'alive'}, index=index)
    expected_mortality_values = pd.Series(current_disease_status, name=disease.cause.name,
                                          index=test_index).map({disease.cause.name: 0.05,
                                                                 f'susceptible_to_{disease.cause.name}': 0})
    disease._mortality.return_value = expected_mortality_values
    rates_df = pd.Series(0, index=test_index, name='death_due_to_other_causes')
    expected = pd.DataFrame({'death_due_to_other_causes': 0, disease.cause.name: expected_mortality_values},
                            index=test_index)

    assert np.all(expected == disease.mortality_rates(test_index, rates_df))


def test_mortality_rate_pandas_dataframe(disease_mock):
    disease = disease_mock('enesmble')
    num_sims = 500
    test_index = range(num_sims)
    current_disease_status = [disease.cause.name] * int(0.2 * num_sims) + \
                             [f'susceptible_to_{disease.cause.name}'] * int(num_sims * 0.8)
    disease.population_view.get.side_effect = lambda index: pd.DataFrame({disease.cause.name: current_disease_status,
                                                                          'alive': 'alive'}, index=index)
    expected_mortality_values = pd.Series(current_disease_status, name=disease.cause.name,
                                          index=test_index).map({disease.cause.name: 0.05,
                                                                 f'susceptible_to_{disease.cause.name}': 0})
    disease._mortality.return_value = expected_mortality_values
    rates_df = pd.DataFrame({'death_due_to_other_causes': 0, 'another_test_cause': 0.001}, index=test_index)
    expected = pd.DataFrame({'death_due_to_other_causes': 0, 'another_test_cause': 0.001,
                             disease.cause.name: expected_mortality_values}, index=test_index)

    assert np.all(expected == disease.mortality_rates(test_index, rates_df))
