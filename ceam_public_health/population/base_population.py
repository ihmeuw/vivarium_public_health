import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_inputs import assign_subregions, get_populations


class BasePopulation:

    def setup(self, builder):
        self._population_data = _add_proportions(
            get_populations(location_id=config.simulation_parameters.location_id))
        self.randomness = builder.randomness('population_generation')

    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive', 'location'])
    def generate_base_population(self, event):
        population_size = len(event.index)
        initial_age = event.user_data.get('initial_age', None)
        sub_pop_data = self._population_data[self._population_data.year == event.time.year]

        population = generate_ceam_population(sub_pop_data, population_size, self.randomness, initial_age=initial_age)
        population.index = event.index
        event.population_view.update(population)


def _add_proportions(population_data):
    def normalize(sub_pop):
        return sub_pop.pop_scaled / sub_pop[sub_pop.sex == 'Both'].pop_scaled.sum()
    population_data['annual_proportion'] = population_data.groupby(
        'year', as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    population_data['annual_proportion_by_age'] = population_data.groupby(
        ['age', 'year'], as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    return population_data


def generate_ceam_population(pop_data, number_of_simulants, randomness_stream, initial_age=None):
    """Returns a population of simulants to be fed into CEAM

    Parameters
    ----------
    pop_data: `DataFrame`
        A table describing the demographic structure of the population to generate.
    number_of_simulants: int
        The size of the population to generate.
    year: int
        Specific year to generate the population for.  Should be the current year in the simulation.
    randomness_stream: `RandomnessStream`
        A randomness source tied to the CRN framework.
    initial_age : int
        If not None simulants will all be set to this age otherwise their
        ages will come from the population distribution

    Returns
    -------
    `DataFrame` :
        Table with columns simulant_id, age, sex, and location.
    """
    simulants = pd.DataFrame({'simulant_id': np.arange(number_of_simulants, dtype=int)})
    if initial_age is not None:
        simulants['age'] = initial_age
        pop_data = pop_data[(pop_data.age_group_start <= initial_age) & (pop_data.age_group_end >= initial_age)]
        # Assign a demographically accurate sex distribution.
        simulants['sex'] = randomness_stream.choice(simulants.index,
                                                    choices=['Male', 'Female'],
                                                    p=[float(pop_data[pop_data.sex == sex].annual_proportion_by_age)
                                                       for sex in ['Male', 'Female']])
    else:
        pop_data = pop_data[pop_data.sex != 'Both']
        pop_data = _rescale_binned_proportions(pop_data)

        choices = pop_data.set_index(['age', 'sex']).annual_proportion.reset_index()
        decisions = randomness_stream.choice(simulants.index,
                                             choices=choices.index,
                                             p=choices.annual_proportion)
        simulants['age'] = choices.loc[decisions, 'age'].values
        simulants['sex'] = choices.loc[decisions, 'sex'].values

    simulants['location'] = pop_data['location_id']
    simulants['alive'] = True
    return simulants


def _rescale_binned_proportions(pop_data):
    pop_age_start = float(config.simulation_parameters.pop_age_start)
    pop_age_end = float(config.simulation_parameters.pop_age_end)
    if pop_age_start is None or pop_age_end is None:
        raise ValueError("Must provide initial_age if pop_age_start and/or pop_age_end are not set.")

    pop_data = pop_data[(pop_data.age_group_start < pop_age_end)
                        & (pop_data.age_group_end > pop_age_start)]

    # TODO: Replace your laziness with a groupby
    for sex in ['Male', 'Female']:
        max_bin = pop_data[(pop_data.age_group_end >= pop_age_end) & (pop_data.sex == sex)]
        min_bin = pop_data[(pop_data.age_group_start <= pop_age_start) & (pop_data.sex == sex)]

        max_scale = float(max_bin.age_group_end)-pop_age_end / float(max_bin.age_group_end - max_bin.age_group_start)
        min_scale = (pop_age_start - float(min_bin.age_group_start)
                     / float(min_bin.age_group_end - min_bin.age_group_start))

        pop_data[pop_data.sex == sex].loc[max_bin.index, 'annual_proportion'] *= max_scale
        pop_data[pop_data.sex == sex].loc[min_bin.index, 'annual_proportion'] *= min_scale

    return pop_data




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
@uses_columns(['age'], 'alive')
def age_simulants(event):
    time_step = config.simulation_parameters.time_step
    event.population['age'] += time_step/365.0
    event.population_view.update(event.population)


