import numpy as np
import pandas as pd
import pytest

from vivarium.framework.utilities import to_yearly
from vivarium.testing_utilities import build_table, TestPopulation
from vivarium.interface import initialize_simulation, setup_simulation

from vivarium_public_health.treatment import HealthcareAccess

np.random.seed(100)


class Metrics:
    def __init__(self):
        self.access_count = 0

    def setup(self, builder):
        builder.event.register_listener('general_healthcare_access', self.count_access)

    def count_access(self, event):
        self.access_count += len(event.index)

    def reset(self):
        self.access_count = 0


@pytest.mark.slow
def test_general_access(base_config, base_plugins):
    year_start = 1990
    year_end = 2005

    step_size = 30.5  # Days, about 1 month
    population_size = 10000
    base_config.update({
        'time': {
            'start': {'year': 1990},
            'end': {'year': 2005},
            'step_size': step_size
        },
        'population': {
            'population_size': population_size
        }
    })
    step_size = pd.Timedelta(days=step_size)
    monthly_rate = 0.1
    annual_rate = to_yearly(monthly_rate, step_size)

    metrics = Metrics()
    simulation = initialize_simulation([TestPopulation(), metrics, HealthcareAccess()],
                                       input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("healthcare_entity.outpatient_visits.annual_visits",
                          build_table(annual_rate, year_start, year_end, ['age', 'year', 'sex', 'annual_visits']))
    simulation.setup()

    steps_to_take = 10 * 12  # ten years
    effective_person_time = population_size * steps_to_take * (step_size/pd.Timedelta(days=365))  # person-years

    simulation.take_steps(steps_to_take)

    effective_annual_rate = metrics.access_count/effective_person_time

    assert abs(annual_rate - effective_annual_rate)/annual_rate < 0.1


@pytest.mark.slow
@pytest.mark.skip("I don't know why this is broken or how it works. -J.C.")
def test_adherence(base_config, get_annual_visits_mock):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    t_step = 28  # days
    n_simulants = int('10_000')

    def get_utilization_rate(*_, **__):
        return build_table(0.1, year_start, year_end, ['age', 'year', 'sex', 'utilization_proportion'])
    get_annual_visits_mock.side_effect = get_utilization_rate

    metrics = Metrics()
    base_config.update({'population': {'population_size': n_simulants},
                        'time': {'step_size': t_step}}, layer='override')
    simulation = setup_simulation([TestPopulation(), metrics, HealthcareAccess()], input_config=base_config)

    simulation.take_steps(number_of_steps=2)

    df = simulation.population.population
    df['fu_visit'] = df.healthcare_visits > 1
    t = df.groupby('adherence_category').fu_visit.count()
    assert t['non-adherent'] == 0, 'non-adherents should not show for follow-up visit'
    assert t['semi-adherent'] < .9*t['adherent'], 'semi-adherents should show up less than adherents for follow-up visit'
