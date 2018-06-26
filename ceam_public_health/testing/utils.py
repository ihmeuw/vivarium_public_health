from itertools import product

import pandas as pd


def make_uniform_pop_data():
    age_bins = [(n, n + 2.5, n + 5) for n in range(0, 100, 5)]
    sexes = ('Male', 'Female', 'Both')
    years = (1990, 1995, 2000, 2005)
    locations = (1, 2)

    age_bins, sexes, years, locations = zip(*product(age_bins, sexes, years, locations))
    mins, ages, maxes = zip(*age_bins)

    pop = pd.DataFrame({'age': ages,
                        'age_group_start': mins,
                        'age_group_end': maxes,
                        'sex': sexes,
                        'year': years,
                        'location': locations,
                        'population': [100] * len(ages)})
    pop.loc[pop.sex == 'Both', 'population'] = 200
    return pop


