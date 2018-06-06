import pytest
import pandas as pd

from vivarium.test_util import TestPopulation, metadata
from vivarium.interface.interactive import setup_simulation

from ceam_public_health.metrics.calculate_incidence import CalculateIncidence
from ceam_public_health.metrics.epidemiology import EpidemiologicalMeasures


@pytest.fixture(scope='function')
def config(base_config):
    base_config.update({
        'time': {
            'start': {'year': 2009},
            'end': {'year': 2011},
            'step_size': 365
        },
        'population': {'population_size': 1000}
    }, **metadata(__file__))

    return base_config


# FIXME: test_calculate_incidence isn't testing anything right now. need to
# figure out how to access the incidence rate value in epidemiological_span_measures
@pytest.mark.skip
def test_calculate_incidence(config):
    factory = diarrhea_factory()
    ci = CalculateIncidence(disease_col='diarrhea', disease='diarrhea', disease_states=['mild_diarrhea'])
    simulation = setup_simulation([TestPopulation(), ci, EpidemiologicalMeasures()] + factory, input_config=config)
    simulation.population.population['diarrhea'] = ['healthy'] * 50 + ['mild_diarrhea'] * 50
    simulation.run_for(pd.Timedelta(days=730))

    inc = simulation.values.get_value('epidemiological_span_measures')

    # assert inc = .5
