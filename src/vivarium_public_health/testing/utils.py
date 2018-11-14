from itertools import product

import pandas as pd


def make_uniform_pop_data(age_bin_midpoint=False):
    age_bins = [(n, n + 5) for n in range(0, 100, 5)]
    sexes = ('Male', 'Female', 'Both')
    years = [(1990, 1995), (1995, 2000), (2000, 2005), (2005, 2010)]
    locations = (1, 2)

    age_bins, sexes, years, locations = zip(*product(age_bins, sexes, years, locations))
    mins, maxes = zip(*age_bins)
    year_starts, year_ends = zip(*years)

    pop = pd.DataFrame({'age_group_start': mins,
                        'age_group_end': maxes,
                        'sex': sexes,
                        'year_start': year_starts,
                        'year_end': year_ends,
                        'location': locations,
                        'population': [100] * len(mins)})
    pop.loc[pop.sex == 'Both', 'population'] = 200
    if age_bin_midpoint:  # used for population tests
        pop['age'] = pop.apply(lambda row: (row['age_group_start'] + row['age_group_end']) / 2, axis=1)
    return pop


