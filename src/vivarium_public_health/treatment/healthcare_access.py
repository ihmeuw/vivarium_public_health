"""
=======================
Healthcare Access Model
=======================

This module contains tools for modeling patient access to healthcare based on
utilization rates.

"""
from collections import defaultdict

import numpy as np
import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.randomness import filter_for_probability


def hospitalization_side_effect_factory(male_probability, female_probability, hospitalization_type):

    class hospitalization_side_effect:

        @property
        def name(self):
            return f'hospitalization_side_effect.{hospitalization_type}'

        def setup(self, builder):
            self.population_view = builder.population.get_view(['sex'])
            self.hospitilization_emitter = builder.event.get_emitter('hospitalization')

        def __call__(self, index, event_time):
            pop = self.population_view.get(index)
            pop['probability'] = 0.0
            pop.loc[pop.sex == 'Male', 'probability'] = male_probability
            pop.loc[pop.sex == 'Female', 'probability'] = female_probability
            effective_population = filter_for_probability('Hospitalization due to {}'.format(hospitalization_type),
                                                          pop.index, pop.probability)
            new_event = Event(effective_population)
            self.hospitilization_emitter(new_event)

    return hospitalization_side_effect()


class HealthcareAccess:

    configuration_defaults = {
        'followup_adherence': {
            # use a dirichlet distribution with means matching Marcia's
            # paper and sum chosen to provide standard deviation on first
            # term also matching paper
            'proportion_adherent': 0.6,
            'proportion_semi_adherent': 0.25,
            'proportion_not_adherent': 0.15,
            # Document this parameter choice, and consider refining it
            'semi_adherent_mean': 0.4,
            'semi_adherent_standard_deviation': 0.0485
        }
    }

    @property
    def name(self):
        return 'healthcare_access'

    def setup(self, builder):
        self.followup_adherence_parameters = builder.configuration.followup_adherence
        self.clock = builder.time.clock()

        utilization = builder.data.load("healthcare_entity.outpatient_visits.utilization")
        self.utilization_rate = builder.value.register_rate_producer('healthcare_utilization.rate',
                                                                     source=builder.lookup.build_table(utilization))
        creates_columns = ['healthcare_last_visit_date', 'healthcare_followup_date']
        self.population_view = builder.population.get_view(['alive'] + creates_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=creates_columns)

        self.randomness_followup_access = builder.randomness.get_stream(self.name + '_followup_access')
        self.randomness_general_access = builder.randomness.get_stream(self.name + 'general_access')
        self.randomness_followup_adherence = builder.randomness.get_stream(self.name + 'followup_adherence')


        builder.event.register_listener('time_step', self.on_time_step)

        self.general_healthcare_access_emitter = builder.event.get_emitter('general_healthcare_access')
        self.followup_healthcare_access_emitter = builder.event.get_emitter('followup_healthcare_access')
        builder.event.register_listener('general_healthcare_access', self.on_healthcare_access)
        builder.event.register_listener('followup_healthcare_access', self.on_healthcare_access)

        self._followup_adherence = pd.Series()
        self.followup_adherence = builder.value.register_value_producer('healthcare_followup.adherence',
                                                                        source=lambda index: self._followup_adherence.loc[index])

    def on_initialize_simulants(self, pop_data):
        self._followup_adherence = self._followup_adherence.append(self.initialize_adherence(pop_data.index))

        self.population_view.update(pd.DataFrame({'healthcare_last_visit_date': pd.NaT,
                                                  'healthcare_followup_date': pd.NaT}, index=pop_data.index))

    def on_time_step(self, event):
        population = self.population_view.get(event.index, query="alive == 'alive'")

        followup_mask = ((population.healthcare_followup_date > self.clock())
                         & (population.healthcare_followup_date <= event.time))
        may_followup_pop = population[followup_mask].index
        to_followup_pop = self.randomness_followup_access.filter_for_probability(may_followup_pop,
                                                                 self.followup_adherence(may_followup_pop))

        may_do_general_access = population.index.difference(may_followup_pop)
        general_access = self.randomness_general_access.filter_for_rate(may_do_general_access,
                                                         self.utilization_rate(population.index))

        self.general_healthcare_access_emitter(Event(general_access))
        self.followup_healthcare_access_emitter(Event(to_followup_pop))

    def on_healthcare_access(self, event):
        self.population_view.update(pd.DataFrame({'healthcare_last_visit_date': event.time}, index=event.index))

    def initialize_adherence(self, index):
        r = np.random.RandomState(self.randomness_followup_adherence.get_seed())
        alpha = np.array([self.followup_adherence_parameters['proportion_adherent'],
                          self.followup_adherence_parameters['proportion_semi_adherent'],
                          self.followup_adherence_parameters['proportion_not_adherent']]) * 100
        p = r.dirichlet(alpha)
        adherence_category = pd.Series(r.choice(['adherent', 'semi-adherent', 'non-adherent'], p=p, size=len(index)),
                                       index=index)

        adherence = pd.Series(1.0, index=index)
        adherence.loc[adherence_category == 'non_adherent'] = 0.0
        adherence.loc[adherence_category == 'semi_adherent'] = r.normal(self.followup_adherence_parameters['semi_adherent_mean'],
                                                                        self.followup_adherence_parameters['semi_adherent_standard_deviation'])
        return adherence

    def __repr__(self):
        return 'HealthcareAccess()'


class HealthcareAccessObserver:

    @property
    def name(self):
        return 'healthcare_access_observer'

    def setup(self, builder):
        ip_cost_df = builder.data.load("healthcare_entity.inpatient_visits.cost")
        op_cost_df = builder.data.load("healthcare_entity.outpatient_visits.cost")
        self.inpatient_cost = builder.lookup.build_table(ip_cost_df)
        self.outpatient_cost = builder.lookup.build_table(op_cost_df)

        self.access_counts = defaultdict(int)  # keys are access type_year, values are counts
        self.cost_by_year = defaultdict(float)  # keys are access type_year, values are cost sums

        builder.event.register_listener('general_healthcare_access', self.on_general_healthcare_access)
        builder.event.register_listener('followup_healthcare_access', self.on_followup_healthcare_access)
        builder.event.register_listener('hospitalization', self.on_hospitalization)

        builder.value.register_value_modifier('metrics', modifier=self.metrics)

    def on_general_healthcare_access(self, event):
        key = f'general_healthcare_access_{event.time.year}'
        self.access_counts[key] += len(event.index)
        self.cost_by_year[key] += self.outpatient_cost(event.index).value.sum()

    def on_followup_healthcare_access(self, event):
        key = f'followup_healthcare_access_{event.time.year}'
        self.access_counts[key] += len(event.index)
        self.cost_by_year[key] += self.outpatient_cost(event.index).value.sum()

    def on_hospitalization(self, event):
        key = f'hospitalization_{event.time.year}'
        self.access_counts[key] += len(event.index)
        self.cost_by_year[key] += self.inpatient_cost(event.index).value.sum()

    def metrics(self, index, metrics):
        for key, value in self.access_counts.items():
            metrics[f'{key}_counts'] = value
        for key, value in self.cost_by_year.items():
            metrics[f'{key}_cost'] = value

        return metrics

    def __repr__(self):
        return "HealthCareAccessObserver()"
