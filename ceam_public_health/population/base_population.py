import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_inputs import (generate_ceam_population, assign_subregions)


@listens_for('initialize_simulants', priority=0)
@uses_columns(['age', 'fractional_age', 'sex', 'alive', 'location'])
def generate_base_population(event):
    population_size = len(event.index)

    # TODO: FIGURE OUT HOW TO SET INITIAL AGE OUTSIDE OF MANUALLY SETTING BELOW
    initial_age = event.user_data.get('initial_age', None)

    population = generate_ceam_population(time=event.time, number_of_simulants=population_size, initial_age=initial_age)
    population['age'] = population.age.astype(int)

    population.index = event.index
    population['fractional_age'] = population.age.astype(float)

    event.population_view.update(population)


@listens_for('initialize_simulants', priority=1)
@uses_columns(['location'])
def assign_location(event):
    main_location = config.simulation_parameters.location_id
    event.population_view.update(assign_subregions(event.index, main_location, event.time.year))

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
    event.population_view.update(pd.Series(r.choice(['adherent', 'semi-adherent', 'non-adherent'], p=p, size=population_size), dtype='category'))

@listens_for('time_step')
@uses_columns(['age', 'fractional_age'], 'alive')
def age_simulants(event):
    time_step = config.simulation_parameters.time_step
    event.population['fractional_age'] += time_step/365.0
    event.population['age'] = event.population.fractional_age.astype(int)
    event.population_view.update(event.population)


