from collections import defaultdict

import numpy as np
import pandas as pd

from vivarium.framework.event import listens_for, emits, Event
from vivarium.framework.randomness import filter_for_probability
from vivarium.interpolation import Interpolation

from ceam_inputs import (get_healthcare_annual_visits, get_inpatient_visit_costs,
                         get_outpatient_visit_costs, healthcare_entities)


def hospitalization_side_effect_factory(male_probability, female_probability, hospitalization_type, population_view):
    @emits('hospitalization')
    def hospitalization_side_effect(index, event_time, emitter):
        pop = population_view.get(index)
        pop['probability'] = 0.0
        pop.loc[pop.sex == 'Male', 'probability'] = male_probability
        pop.loc[pop.sex == 'Female', 'probability'] = female_probability
        effective_population = filter_for_probability('Hospitalization due to {}'.format(hospitalization_type),
                                                      pop.index, pop.probability)
        new_event = Event(effective_population)
        emitter(new_event)
    return hospitalization_side_effect


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
        self.clock = builder.clock()
        r = np.random.RandomState(self.general_random.get_seed())

        self.semi_adherent_pr = r.normal(0.4, 0.0485)  # FIXME: document this parameter choice, and consider refining it

        self.cost_by_year = defaultdict(float)
        self.general_access_count = 0
        self.followup_access_count = 0
        self.hospitalization_count = 0

        self.hospitalization_cost = defaultdict(float)
        ip_cost_df = get_inpatient_visit_costs(builder.configuration).rename(columns={'year_id': 'year'})
        self._hospitalization_cost = Interpolation(ip_cost_df, tuple(), ('year',))

        cost_df = get_outpatient_visit_costs(builder.configuration)
        self._appointment_cost = Interpolation(cost_df, tuple(), ('year',))

        self.outpatient_cost = defaultdict(float)

        self.general_healthcare_access_emitter = builder.emitter('general_healthcare_access')
        self.followup_healthcare_access_emitter = builder.emitter('followup_healthcare_access')

        annual_visits = get_healthcare_annual_visits(healthcare_entities.outpatient_visits, builder.configuration)
        self.utilization_rate = builder.value.register_rate_producer('healthcare_utilization.rate',
                                                                     source=builder.lookup(annual_visits))
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

        self.population_view = builder.population_view(['healthcare_followup_date', 'healthcare_last_visit_date',
                                                        'healthcare_visits', 'adherence_category',
                                                        'general_access_propensity'])

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        population_size = len(event.index)
        adherence = self.get_adherence(population_size)

        r = np.random.RandomState(self.general_random.get_seed())
        general_access_propensity = np.ones(shape=population_size) #r.uniform(size=population_size)**3

        # normalize propensity to have mean 1, so it can be multiplied
        # in without changing population mean rate
        general_access_propensity /= general_access_propensity.mean()

        self.population_view.update(pd.DataFrame({'healthcare_followup_date': [pd.NaT]*population_size,
                                                  'healthcare_last_visit_date': [pd.NaT]*population_size,
                                                  'healthcare_visits': [0]*population_size,
                                                  'adherence_category': adherence,
                                                  'general_access_propensity': general_access_propensity}))

    def get_adherence(self, population_size):
        # use a dirichlet distribution with means matching Marcia's
        # paper and sum chosen to provide standard deviation on first
        # term also matching paper
        r = np.random.RandomState(self.adherence_random.get_seed())
        alpha = np.array([0.6, 0.25, 0.15]) * 100
        p = r.dirichlet(alpha)
        return pd.Series(r.choice(['adherent', 'semi-adherent', 'non-adherent'], p=p, size=population_size),
                         dtype='category')

    @listens_for('time_step')
    def general_access(self, event):
        population = self.population_view.get(event.index, query="alive == 'alive")
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
        event.population_view.update(population.healthcare_visits)

        year = event.time.year
        self.cost_by_year[year] += len(index) * self._appointment_cost(year=[year])[0]
        self.outpatient_cost[year] += len(index) * self._appointment_cost(year=[year])[0]

    @listens_for('time_step')
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
        self.cost_by_year[year] += len(affected_population) * self._appointment_cost(year=[year])[0]
        self.outpatient_cost[year] += len(affected_population) * self._appointment_cost(year=[year])[0]

    @listens_for('hospitalization')
    def hospital_access(self, event):
        year = event.time.year
        self.hospitalization_count += len(event.index)
        self.hospitalization_cost[year] += len(event.index) * self._hospitalization_cost(year=[year])[0]
        self.cost_by_year[year] += len(event.index) * self._hospitalization_cost(year=[year])[0]

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
