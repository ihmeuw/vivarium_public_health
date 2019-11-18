"""
=================
Testing Utilities
=================

This module contains data generation tools for testing vivarium_public_health
components.

"""
from itertools import product

import pandas as pd


def make_uniform_pop_data(age_bin_midpoint=False):
    age_bins = [(n, n + 5) for n in range(0, 100, 5)]
    sexes = ('Male', 'Female')
    years = zip(range(1990, 2018), range(1991, 2019))
    locations = (1, 2)

    age_bins, sexes, years, locations = zip(*product(age_bins, sexes, years, locations))
    mins, maxes = zip(*age_bins)
    year_starts, year_ends = zip(*years)

    pop = pd.DataFrame({'age_start': mins,
                        'age_end': maxes,
                        'sex': sexes,
                        'year_start': year_starts,
                        'year_end': year_ends,
                        'location': locations,
                        'value': [100] * len(mins)})
    if age_bin_midpoint:  # used for population tests
        pop['age'] = pop.apply(lambda row: (row['age_start'] + row['age_end']) / 2, axis=1)
    return pop
