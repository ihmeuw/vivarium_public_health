import numpy as np
import pandas as pd
import pytest
from vivarium_public_health.disease import RiskAttributableDisease


@pytest.fixture
def disease_mock(mocker):
    def disease_with_distribution(distribution):
        test_disease = RiskAttributableDisease('test_cause', 'test_risk')
        test_disease.exposure = mocker.Mock()
        test_disease.distribution = distribution
        test_disease.exposure = mocker.Mock()
        test_disease._get_population = mocker.Mock()
        test_disease._mortality = mocker.Mock()
        return test_disease
    return disease_with_distribution


test_data = [('ordered_polytomous', ['cat1', 'cat2', 'cat3', 'cat4'], ['cat1']),
             ('ordered_polytomous', ['cat1', 'cat2', 'cat3', 'cat4'], ['cat1', 'cat2']),
             ('ordered_polytomous', ['cat1', 'cat2', 'cat3', 'cat4'], ['cat1', 'cat2', 'cat3']),
             ('dichotomous', ['cat1', 'cat2'], ['cat1'])]


@pytest.mark.parametrize('distribution, categories, threshold', test_data)
def test_filter_by_exposure_categorical(disease_mock, distribution, categories, threshold):
    disease = disease_mock(distribution)
    disease.threshold = threshold
    test_index = range(500)
    per_cat = len(test_index) // len(categories)
    infected = threshold * per_cat
    susceptible = list(set(categories)-set(threshold)) * per_cat

    current_exposure = lambda index: pd.Series(infected + susceptible, index=index)
    expected = lambda index: current_exposure(index).isin(infected)
    disease.exposure.side_effect = current_exposure

    assert np.all(expected(test_index) == disease.filter_by_exposure(test_index))


test_data =[('ensemble', 7), ('lognormal', 2.5), ('normal', 4)]


@pytest.mark.parametrize('distribution, threshold', test_data)
def test_filter_by_exposure_continuous(disease_mock, distribution, threshold):
    disease = disease_mock(distribution)
    disease.threshold = threshold
    test_index= range(500)
    current_exposure = lambda index: pd.Series([threshold - 0.2, threshold - 0.1, threshold, threshold + 0.1,
                                                threshold + 0.2] *100, index=test_index)
    expected = lambda index: current_exposure(index).isin([threshold + 0.1, threshold + 0.2])
    disease.exposure.side_effect = current_exposure

    assert np.all(expected(test_index) == disease.filter_by_exposure(test_index))


def test_mortality_rate_pandas_series(disease_mock):
    disease = disease_mock('enesmble')
    num_sims = 500
    test_index = range(num_sims)
    current_disease_status = [disease.name] * int(0.2 * num_sims) + \
                             [f'susceptible_to_{disease.name}'] * int(num_sims * 0.8)
    disease._get_population.return_value = pd.DataFrame({disease.name: current_disease_status, 'alive': 'alive'},
                                                        index=test_index)
    expected_mortality_values = pd.Series(current_disease_status, name=disease.name,
                                          index=test_index).map({disease.name: 0.05, f'susceptible_to_{disease.name}':0})
    disease._mortality.return_value = expected_mortality_values
    rates_df = pd.Series(0, index=test_index, name='death_due_to_other_causes')
    expected = pd.DataFrame({'death_due_to_other_causes': 0, disease.name: expected_mortality_values}, index=test_index)

    assert np.all(expected == disease.mortality_rates(test_index, rates_df))


def test_mortality_rate_pandas_dataframe(disease_mock):
    disease = disease_mock('enesmble')
    num_sims = 500
    test_index = range(num_sims)
    current_disease_status = [disease.name] * int(0.2 * num_sims) + \
                             [f'susceptible_to_{disease.name}'] * int(num_sims * 0.8)
    disease._get_population.return_value = pd.DataFrame({disease.name: current_disease_status, 'alive': 'alive'},
                                                        index=test_index)
    expected_mortality_values = pd.Series(current_disease_status, name=disease.name,
                                          index=test_index).map({disease.name: 0.05, f'susceptible_to_{disease.name}':0})
    disease._mortality.return_value = expected_mortality_values
    rates_df = pd.DataFrame({'death_due_to_other_causes': 0, 'another_test_cause': 0.001}, index=test_index)
    expected = pd.DataFrame({'death_due_to_other_causes': 0, 'another_test_cause': 0.001,
                             disease.name: expected_mortality_values}, index=test_index)

    assert np.all(expected == disease.mortality_rates(test_index, rates_df))