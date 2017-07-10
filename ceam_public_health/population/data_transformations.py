import numpy as np
import pandas as pd

from ceam import config
from ceam_inputs import get_subregions, get_populations


def add_proportions(population_data):
    def normalize(sub_pop):
        return sub_pop.pop_scaled / sub_pop[sub_pop.sex == 'Both'].pop_scaled.sum()
    population_data['annual_proportion'] = population_data.groupby(
        'year', as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    population_data['annual_proportion_by_age'] = population_data.groupby(
        ['age', 'year'], as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    return population_data


def generate_ceam_population(pop_data, number_of_simulants, randomness_stream, initial_age=None):
    simulants = pd.DataFrame({'simulant_id': np.arange(number_of_simulants, dtype=int),
                              'alive': ['alive']*number_of_simulants})
    if initial_age is not None:
        simulants['age'] = float(initial_age)
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
        # TODO: Smooth out ages.
        simulants['age'] = choices.loc[decisions, 'age'].values
        simulants['sex'] = choices.loc[decisions, 'sex'].values
        simulants = _smooth_ages(simulants, pop_data)

    return simulants


def assign_subregions(index, location, year, randomness):
    sub_regions = get_subregions(location)

    # TODO: Use demography in a smart way here.
    if sub_regions:
        sub_pops = np.array([get_populations(sub_region, year=year, sex='Both').pop_scaled.sum()
                             for sub_region in sub_regions])
        proportions = sub_pops / sub_pops.sum()
        return randomness.choice(index=index, choices=sub_regions, p=proportions)
    else:
        return pd.Series(location, index=index)


def _rescale_binned_proportions(pop_data):
    pop_age_start = float(config.simulation_parameters.pop_age_start)
    pop_age_end = float(config.simulation_parameters.pop_age_end)
    if pop_age_start is None or pop_age_end is None:
        raise ValueError("Must provide initial_age if pop_age_start and/or pop_age_end are not set.")

    pop_data = pop_data[(pop_data.age_group_start < pop_age_end)
                        & (pop_data.age_group_end > pop_age_start)]

    for sex in ['Male', 'Female']:
        max_bin = pop_data[(pop_data.age_group_end >= pop_age_end) & (pop_data.sex == sex)]
        min_bin = pop_data[(pop_data.age_group_start <= pop_age_start) & (pop_data.sex == sex)]

        max_scale = float(max_bin.age_group_end)-pop_age_end / float(max_bin.age_group_end - max_bin.age_group_start)
        min_scale = (pop_age_start - float(min_bin.age_group_start)
                     / float(min_bin.age_group_end - min_bin.age_group_start))

        pop_data[pop_data.sex == sex].loc[max_bin.index, 'annual_proportion'] *= max_scale
        pop_data[pop_data.sex == sex].loc[min_bin.index, 'annual_proportion'] *= min_scale

    return pop_data


def _smooth_ages(simulants, pop_data):
    pop_data['cdf'] = _compute_cdf(pop_data)
    return simulants


def _compute_cdf(pop_data):
    return pd.Series(0, index=pop_data.index)
