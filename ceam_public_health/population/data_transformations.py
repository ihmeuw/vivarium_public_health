import numpy as np


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

    if pop_age_start != pop_data.age_group_start.min():
        pop_data = pop_data[pop_data.age_group_end > pop_age_start]
    if pop_age_end != pop_data.age_group_end.max():
        pop_data = pop_data[pop_data.age_group_start < pop_age_end]

    for sex in ['Male', 'Female']:
        max_bin = pop_data[(pop_data.age_group_end >= pop_age_end) & (pop_data.sex == sex)]
        min_bin = pop_data[(pop_data.age_group_start <= pop_age_start) & (pop_data.sex == sex)]

        max_scale = float(max_bin.age_group_end)-pop_age_end / float(max_bin.age_group_end - max_bin.age_group_start)
        min_scale = (pop_age_start - float(min_bin.age_group_start)
                     / float(min_bin.age_group_end - min_bin.age_group_start))

        pop_data[pop_data.sex == sex].loc[max_bin.index, 'annual_proportion'] *= max_scale
        pop_data[pop_data.sex == sex].loc[min_bin.index, 'annual_proportion'] *= min_scale

    return pop_data


def smooth_ages(simulants, population_data, randomness):
    for sex in ['Male', 'Female']:
        pop_data = population_data[population_data.sex == sex]

        ages = sorted(pop_data.age.unique())
        younger = [0] + ages[:-1]
        older = ages[1:] + [float(pop_data.loc[pop_data.age == ages[-1], 'age_group_end'])]
        uniform_all = randomness.get_draw(simulants.index, additional_key='smooth_ages')
        for age, young, old in zip(ages, younger, older):
            affected = simulants[(simulants.age == age) & (simulants.sex == sex)]
            # bin endpoints
            left = float(pop_data.loc[pop_data.age == age, 'age_group_start'])
            right = float(pop_data.loc[pop_data.age == age, 'age_group_end'])

            # proportion in this bin and the neighboring bins
            p_age = float(pop_data.loc[pop_data.age == age, 'annual_proportion'])
            p_young = float(pop_data.loc[pop_data.age == young, 'annual_proportion']) if young != left else p_age
            p_old = float(pop_data.loc[pop_data.age == old, 'annual_proportion']) if old != right else 0

            # pdf value at bin endpoints
            f_left = (p_age - p_young)/(age - young)*(left - young) + p_young
            f_right = (p_old - p_age)/(old - age)*(right - age) + p_age

            # normalization constant.  Total area under pdf.
            area = 0.5*((p_age + f_left)*(age - left) + (f_right + p_age)*(right - age))

            # pdf slopes.
            m_left = (p_age - f_left)/(age - left)
            m_right = (f_right - p_age)/(right - age)

            # The decision bound on the uniform rv.
            cdf_at_age = 1/(2*area)*(p_age + f_left)*(age - left)

            # Make a draw from a uniform distribution
            uniform_rv = uniform_all.iloc[affected.index]

            left_sims = affected[uniform_rv <= cdf_at_age]
            right_sims = affected[uniform_rv > cdf_at_age]

            # Compute and assign ages.
            if m_left == 0:
                simulants.loc[left_sims.index, 'age'] = left + area/f_left*uniform_rv[left_sims.index]
            else:
                simulants.loc[left_sims.index, 'age'] = left + f_left/m_left*(
                    np.sqrt(1 + 2*area*m_left/f_left**2*uniform_rv[left_sims.index]) - 1)

            if m_right == 0:
                simulants.loc[right_sims.index, 'age'] = age + area/p_age*(uniform_rv[right_sims.index] - cdf_at_age)
            else:
                simulants.loc[right_sims.index, 'age'] = age + p_age/m_right*(
                    np.sqrt(1 + 2*area*m_right/p_age**2*(uniform_rv[right_sims.index] - cdf_at_age)) - 1)

    return simulants



