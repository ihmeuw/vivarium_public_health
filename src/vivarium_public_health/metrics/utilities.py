from string import Template

import pandas as pd


def get_age_bins(builder):
    age_bins = builder.data.load('population.age_bins')
    exit_age = builder.configuration.population.exit_age
    if exit_age:
        age_bins = age_bins[age_bins.age_group_start < exit_age]
        age_bins.loc[age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age
    return age_bins


def get_output_template(by_age, by_sex, by_year):
    template = '{measure}'
    if by_year:
        template += '_in_{year}'
    if by_sex:
        template += '_among_{sex}'
    if by_age:
        template += 'in_age_group_{age_group}'
    return Template(template)


def get_group_counts(pop, base_filter, base_key, config, age_bins):
    if config.by_age:
        ages = age_bins.iterrows()
        base_filter += ' and ({age_group_start} <= age) and (age < {age_group_end})'
    else:
        ages = [('all_ages', pd.Series({'age_group_start': None, 'age_group_end': None}))]

    if config.by_sex:
        sexes = ['Male', 'Female']
        base_filter += ' and sex == {sex}'
    else:
        sexes = ['Both']

    group_counts = {}

    for group, age_group in ages:
        start, end = age_group.age_group_start, age_group.age_group_end
        for sex in sexes:
            filter_kwargs = {'age_group_start': start, 'age_group_end': end, 'sex': sex}
            key = base_key.safe_substitute(**filter_kwargs)
            group_filter = base_filter.format(**filter_kwargs)

            in_group = pop.query(group_filter)

            group_counts[key] = len(in_group)

    return group_counts


def clean_cause_of_death(pop):

    def _clean(cod):
        if 'death' in cod or 'dead' in cod:
            pass
        else:
            cod = f'death_due_to_{cod}'
        return cod

    pop.cause_of_death = pop.cause_of_death.apply(_clean)
    return pop


def to_years(time) -> float:
    return time / pd.Timedelta(days=365.25)
