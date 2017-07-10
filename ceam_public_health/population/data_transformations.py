import numpy as np
import pandas as pd


def add_proportions(population_data):
    def normalize(sub_pop):
        return sub_pop.pop_scaled / sub_pop[sub_pop.sex == 'Both'].pop_scaled.sum()
    population_data['annual_proportion'] = population_data.groupby(
        'year', as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    population_data['annual_proportion_by_age'] = population_data.groupby(
        ['age', 'year'], as_index=False).apply(normalize).reset_index(level=0).pop_scaled
    return population_data


def rescale_binned_proportions(pop_data, pop_age_start, pop_age_end):
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


def smooth_ages(simulants, pop_data):
    pop_data['cdf'] = compute_cdf(pop_data)
    return simulants


def compute_cdf(pop_data):
    return pd.Series(0, index=pop_data.index)
