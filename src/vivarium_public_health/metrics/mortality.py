from string import Template

import pandas as pd
import numpy as np

from .utilities import get_age_bins, clean_cause_of_death, to_years, get_output_template, QueryString, get_group_counts


class MortalityObserver:
    """ An observer for total and cause specific deaths during simulation.
    This component counts total and cause specific deaths in the population
    as well as person time (the time spent alive and tracked in the
    simulation).
    The data is discretized by age groups and optionally by year.
    """
    configuration_defaults = {
        'metrics': {
            'mortality': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
                'among_born': False,
            }
        }
    }

    def setup(self, builder):
        self.name = 'mortality_observer'

        self.config = builder.configuration.metrics.mortality

        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.start_time = self.clock()
        self.initial_pop_entrance_time = self.start_time - self.step_size()

        self.output_template = get_output_template(**self.config.to_dict())

        self.age_bins = get_age_bins(builder)

        columns_required = ['tracked', 'alive', 'entrance_time', 'exit_time', 'cause_of_death', 'years_of_life_lost']
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']

        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()
        pop = clean_cause_of_death(pop)

        person_time = get_person_time(pop, self.output_template, self.config,
                                      self.start_time, self.clock(), self.age_bins)
        metrics.update(person_time)

        deaths = get_deaths(pop, self.output_template, self.config,
                            self.start_time, self.clock(), self.age_bins)
        metrics.update(deaths)

        return metrics


def get_person_time(pop: pd.DataFrame, base_key: Template,
                    config: dict, sim_start: pd.Timestamp, sim_end: pd.Timestamp,
                    age_bins: pd.DataFrame) -> dict:
    base_filter = QueryString('')
    base_key = base_key.safe_substitute(measure='person_time')

    if config['by_age']:
        ages = age_bins.iterrows()
        base_filter += '({age_group_start} <= age) and (age_at_start < {age_group_end})'
    else:
        ages = [('all_ages', pd.Series({'age_group_start': None, 'age_group_end': None}))]

    if config['by_sex']:
        sexes = ['Male', 'Female']
        base_filter += 'sex == {sex}'
    else:
        sexes = ['Both']

    if config['by_year']:
        years = [(year, (pd.Timestamp(f'1-1-{year}'), pd.Timestamp(f'1-1-{year + 1}')))
                 for year in range(sim_start.year, sim_end.year + 1)]
    else:
        years = [('all_years', (pd.Timestamp(f'1-1-1000'), pd.Timestamp(f'1-1-5000')))]
    # This filter needs to be applied separately to compute additional
    # attributes in the person time calculation.
    span_filter = '{t_start} <= exit_time and entrance_time < {t_end}'

    out = {}
    for year, (t_start, t_end) in years:
        lived_in_span = pop.query(span_filter.format(t_start, t_end))

        entrance_time = lived_in_span.entrance_time
        exit_time = lived_in_span.exit_time
        exit_time.loc[t_end < exit_time] = t_end

        years_in_span = to_years(exit_time - entrance_time)
        lived_in_span['age_at_start'] = np.maximum(lived_in_span.age - years_in_span, 0)

        for sex in sexes:
            for group, age_bin in ages:
                a_start, a_end = age_bin.age_group_start, age_bin.age_group_end
                filter_kwargs = {'year': year, 'sex': sex, 'age_group_start': a_start, 'age_group_end': a_end}

                group_filter = base_filter.format(**filter_kwargs)
                in_group = lived_in_span.query(group_filter) if group_filter else lived_in_span.copy()
                age_start = np.maximum(in_group.age_at_start, a_start)
                age_end = np.minimum(in_group.age, a_end)

                key = base_key.substitute(**filter_kwargs)

                out[key] = (age_end - age_start).sum()

    return out


def get_deaths(pop: pd.DataFrame, base_key: Template, config: dict,
               sim_start: pd.Timestamp, sim_end: pd.Timestamp, age_bins: pd.DataFrame):
    base_filter = QueryString('alive == "dead"')

    if config['by_year']:
        years = [(year, (pd.Timestamp(f'1-1-{year}'), pd.Timestamp(f'1-1-{year + 1}')))
                 for year in range(sim_start.year, sim_end.year + 1)]
    else:
        years = [('all_years', (pd.Timestamp(f'1-1-1000'), pd.Timestamp(f'1-1-5000')))]
    additional_filter = '{t_start} <= exit_time and entrance_time < {t_end}'

    causes = [c for c in pop.cause_of_death.unique()]
    additional_filter += ' and cause_of_death == {cause}'

    deaths = {}
    for cause, year, (t_start, t_end) in zip(causes, years):
        cause_year_filter = base_filter + additional_filter.format(t_start, t_end, cause)
        group_deaths = get_group_counts(pop, cause_year_filter, base_key, config, age_bins)

        for key, count in group_deaths:
            key = key.substitute(measure=cause, year=year)
            deaths[key] = count

    return deaths
