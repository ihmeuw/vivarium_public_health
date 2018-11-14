from collections import defaultdict

import numpy as np
import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.randomness import filter_for_probability


def hospitalization_side_effect_factory(male_probability, female_probability, hospitalization_type):

    class hospitalization_side_effect:
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
    """Model health care utilization.

    This includes access events due to chance (broken arms, flu, etc.) and those due to follow up
    appointments, which are affected by adherence rate. This module does not schedule
    follow-up visits.  But it implements the response to follow-ups added to the `healthcare_followup_date`
    column by other modules (for example opportunistic_screening.OpportunisticScreeningModule).

    Population Columns
    ------------------
    healthcare_last_visit_date : pd.Timestamp
        most recent health care access event

    healthcare_followup_date : pd.Timestamp
        next scheduled follow-up appointment
    """

    configuration_defaults = {
            'appointments': {
                'cost': 7.29,
            }
    }

    def setup(self, builder):
        self.general_random = builder.randomness.get_stream('healthcare_general_access')
        self.followup_random = builder.randomness.get_stream('healthcare_followup_access')
        self.adherence_random = builder.randomness.get_stream('healthcare_adherence')
        self.clock = builder.time.clock()
        r = np.random.RandomState(self.general_random.get_seed())

        self.semi_adherent_pr = r.normal(0.4, 0.0485)  # FIXME: document this parameter choice, and consider refining it

        self.cost_by_year = defaultdict(float)
        self.general_access_count = 0
        self.followup_access_count = 0
        self.hospitalization_count = 0

        interpolation_order = builder.configuration.interpolation.order
        self.hospitalization_cost = defaultdict(float)
        self._ip_cost_df = builder.data.load("healthcare_entity.inpatient_visits.cost")
        self.__hospitalization_cost = builder.lookup.build_table(self._ip_cost_df[['year_start', 'year_end', 'value']],
                                                                 tuple(), [('year', 'year_start', 'year_end')])

        self._op_cost_df = builder.data.load("healthcare_entity.outpatient_visits.cost")
        self.__appointment_cost = builder.lookup.build_table(self._op_cost_df[['year_start', 'year_end', 'value']],
                                                             tuple(), [('year', 'year_start', 'year_end')])

        self.outpatient_cost = defaultdict(float)

        self.general_healthcare_access_emitter = builder.event.get_emitter('general_healthcare_access')
        self.followup_healthcare_access_emitter = builder.event.get_emitter('followup_healthcare_access')

        annual_visits = builder.data.load("healthcare_entity.outpatient_visits.annual_visits")
        self.utilization_rate = builder.value.register_rate_producer('healthcare_utilization.rate',
                                                                     source=builder.lookup.build_table(annual_visits))
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

        columns = ['healthcare_followup_date', 'healthcare_last_visit_date', 'healthcare_visits',
                   'adherence_category', 'general_access_propensity']
        self.population_view = builder.population.get_view(columns)
        builder.population.initializes_simulants(self.load_population_columns, creates_columns=columns)

        builder.event.register_listener('time_step', self.general_access)
        builder.event.register_listener('time_step', self.followup_access)
        builder.event.register_listener('hospitalization', self.hospital_access)

    
    @property
    def _hospitalization_cost(self):
        return self.__hospitalization_cost

    @property
    def _appointment_cost(self):
        return self.__appointment_cost

    def load_population_columns(self, pop_data):
        population_size = len(pop_data.index)
        adherence = self.get_adherence(population_size)

        r = np.random.RandomState(self.general_random.get_seed())
        general_access_propensity = np.ones(shape=population_size)  # r.uniform(size=population_size)**3

        # normalize propensity to have mean 1, so it can be multiplied
        # in without changing population mean rate
        general_access_propensity /= general_access_propensity.mean()

        self.population_view.update(pd.DataFrame({'healthcare_followup_date': [pd.NaT]*population_size,
                                                  'healthcare_last_visit_date': [pd.NaT]*population_size,
                                                  'healthcare_visits': [0]*population_size,
                                                  'adherence_category': adherence,
                                                  'general_access_propensity': general_access_propensity},
                                                 index=pop_data.index))

    def get_adherence(self, population_size):
        # use a dirichlet distribution with means matching Marcia's
        # paper and sum chosen to provide standard deviation on first
        # term also matching paper
        r = np.random.RandomState(self.adherence_random.get_seed())
        alpha = np.array([0.6, 0.25, 0.15]) * 100
        p = r.dirichlet(alpha)
        return pd.Series(r.choice(['adherent', 'semi-adherent', 'non-adherent'], p=p, size=population_size),
                         dtype='category')

    def general_access(self, event):
        population = self.population_view.get(event.index, query="alive == 'alive'")
        # determine population who accesses care
        t = self.utilization_rate(event.index)

        # scale based on general access propensity
        t *= population.general_access_propensity

        index = self.general_random.filter_for_rate(event.index, t)

        # for those who show up, emit_event that the visit has happened, and tally the cost
        self.population_view.update(pd.Series(event.time, index=index, name='healthcare_last_visit_date'))
        self.general_healthcare_access_emitter(event.split(index))
        self.general_access_count += len(index)

        population.healthcare_visits += 1
        self.population_view.update(population.healthcare_visits)

        year = event.time.year
        self.cost_by_year[year] += self._appointment_cost(index).sum()
        self.outpatient_cost[year] += self._appointment_cost(index).sum()

    def followup_access(self, event):
        # determine population due for a follow-up appointment
        population = self.population_view.get(event.index, query="alive == 'alive'")
        rows = ((population.healthcare_followup_date > self.clock())
                & (population.healthcare_followup_date <= event.time))
        affected_population = population[rows]

        # of them, determine who shows up for their follow-up appointment
        adherence_pr = {'adherent': 1.0,
                        'semi-adherent': self.semi_adherent_pr,
                        'non-adherent': 0.0}
        adherence = affected_population.adherence_category.map(adherence_pr)

        affected_population = self.followup_random.filter_for_probability(affected_population, adherence)

        # for those who show up, emit_event that the visit has happened, and tally the cost
        self.population_view.update(pd.Series(event.time, index=affected_population.index,
                                              name='healthcare_last_visit_date'))
        self.followup_healthcare_access_emitter(event.split(affected_population.index))
        self.followup_access_count += len(affected_population)

        population.healthcare_visits += 1
        self.population_view.update(population.healthcare_visits)

        year = event.time.year
        self.cost_by_year[year] += self._appointment_cost(affected_population.index).sum()
        self.outpatient_cost[year] += self._appointment_cost(affected_population.index).sum()

    def hospital_access(self, event):
        year = event.time.year
        self.hospitalization_count += len(event.index)
        self.hospitalization_cost[year] += self._hospitalization_cost(event.index).sum()
        self.cost_by_year[year] +=  self._hospitalization_cost(event.index).sum()

    def metrics(self, index, metrics):
        metrics['healthcare_access_cost'] = sum(self.cost_by_year.values())
        metrics['general_healthcare_access'] = self.general_access_count
        metrics['followup_healthcare_access'] = self.followup_access_count
        metrics['hospitalization_access'] = self.hospitalization_count
        metrics['hospitalization_cost'] = sum(self.hospitalization_cost.values())
        metrics['outpatient_cost'] = sum(self.outpatient_cost.values())

        if 'cost' in metrics:
            metrics['cost'] += metrics['healthcare_access_cost']
        else:
            metrics['cost'] = metrics['healthcare_access_cost']
        return metrics
