import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_inputs import get_populations
from .data_transformations import add_proportions, generate_ceam_population, assign_subregions


class BasePopulation:

    def setup(self, builder):
        self._population_data = add_proportions(
            get_populations(location_id=config.simulation_parameters.location_id))
        self.randomness = builder.randomness('population_generation')

    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive', 'location', 'entrance_time', 'exit_time'])
    def generate_base_population(self, event):
        population_size = len(event.index)
        initial_age = event.user_data.get('initial_age', None)
        sub_pop_data = self._population_data[self._population_data.year == event.time.year]

        population = generate_ceam_population(sub_pop_data, population_size, self.randomness, initial_age=initial_age)
        population.index = event.index
        population['entrance_time'] = pd.Timestamp(event.time)
        population['exit_time'] = pd.NaT
        event.population_view.update(population)

    @listens_for('initialize_simulants', priority=1)
    @uses_columns(['location'])
    def assign_location(self, event):
        main_location = config.simulation_parameters.location_id
        if 'use_subregions' in config.simulation_parameters and config.simulation_parameters.use_subregions:
            event.population_view.update(assign_subregions(index=event.index, location=main_location,
                                                           year=event.time.year, randomness=self.randomness))
        else:
            event.population_view.update(pd.Series(main_location, index=event.index))


@listens_for('initialize_simulants')
@uses_columns(['adherence_category'])
def adherence(event):
    population_size = len(event.index)
    # use a dirichlet distribution with means matching Marcia's
    # paper and sum chosen to provide standard deviation on first
    # term also matching paper
    draw_number = config.run_configuration.draw_number
    r = np.random.RandomState(1234567+draw_number)
    alpha = np.array([0.6, 0.25, 0.15]) * 100
    p = r.dirichlet(alpha)
    # then use these probabilities to generate adherence
    # categories for all simulants
    event.population_view.update(pd.Series(r.choice(['adherent', 'semi-adherent', 'non-adherent'],
                                                    p=p, size=population_size), dtype='category'))


@listens_for('time_step')
@uses_columns(['age'], "alive == 'alive'")
def age_simulants(event):
    time_step = config.simulation_parameters.time_step
    event.population['age'] += time_step/365.0
    event.population_view.update(event.population)


@listens_for('time_step', priority=1)  # Set slightly after mortality.
@uses_columns(['alive', 'age', 'exit_time'], "alive == 'alive'")
def age_out_simulants(event):
    if 'maximum_age' not in config.simulation_parameters:
        raise ValueError('Must specify a maximum age in the config in order to use this component.')
    max_age = float(config.simulation_parameters.maximum_age)
    pop = event.population[event.population['age'] >= max_age].copy()
    pop['alive'] = 'untracked'
    pop['age'] = max_age
    pop['exit_time'] = pd.Timestamp(event.time)
    event.population_view.update(pop)


