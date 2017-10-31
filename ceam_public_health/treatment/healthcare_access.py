from collections import defaultdict

import numpy as np
import pandas as pd

from vivarium.framework.event import listens_for, emits, Event
from vivarium.framework.population import uses_columns
from vivarium.framework.values import modifies_value
from vivarium.framework.randomness import filter_for_probability

from ceam_inputs import get_proportion, get_inpatient_visit_costs, get_outpatient_visit_costs, healthcare_entities
from ceam_public_health.util import simple_extrapolation


def hospitalization_side_effect_factory(male_probability, female_probability, hospitalization_type):
    @emits('hospitalization')
    @uses_columns(['sex'])
    def hospitalization_side_effect(index, event_time, emitter, population_view):
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
        location_id = builder.configuration.input_data.location_id
        draw = builder.configuration.run_configuration.input_draw_number

        utilization_data = get_proportion(healthcare_entities.outpatient_visits, builder.configuration)

        self.general_random = builder.randomness('healthcare_general_access')
        self.followup_random = builder.randomness('healthcare_followup_access')
        self.adherence_random = builder.randomness('healthcare_adherence')
        self.clock = builder.clock()
        r = np.random.RandomState(self.general_random.get_seed())

        self.semi_adherent_pr = r.normal(0.4, 0.0485)

        self.cost_by_year = defaultdict(float)
        self.general_access_count = 0
        self.followup_access_count = 0
        self.hospitalization_count = 0

        self.hospitalization_cost = defaultdict(float)
        ip_cost_df = get_inpatient_visit_costs(builder.configuration)
        ip_cost_df.index = ip_cost_df.year_id
        self._hospitalization_cost = simple_extrapolation(ip_cost_df['draw_{}'.format(draw)])

        cost_df = get_outpatient_visit_costs(builder.configuration).set_index('year_id')
        self._appointment_cost = simple_extrapolation(cost_df.loc[cost_df.location_id == location_id,
                                                                       'draw_{}'.format(draw)])

        self.outpatient_cost = defaultdict(float)

        self.general_healthcare_access_emitter = builder.emitter('general_healthcare_access')
        self.followup_healthcare_access_emitter = builder.emitter('followup_healthcare_access')


        self.utilization_proportion = builder.lookup(utilization_data)

    @listens_for('initialize_simulants')
    @uses_columns(['healthcare_followup_date', 'healthcare_last_visit_date', 'adherence_category'])
    def load_population_columns(self, event):
        population_size = len(event.index)
        adherence = self.get_adherence(population_size)
        event.population_view.update(pd.DataFrame({'healthcare_followup_date': [pd.NaT]*population_size,
                                                   'healthcare_last_visit_date': [pd.NaT]*population_size,
                                                   'adherence_category': adherence}))

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
    @uses_columns(['healthcare_last_visit_date'], "alive == 'alive'")
    def general_access(self, event):
        # determine population who accesses care
        t = self.utilization_proportion(event.index)
        # FIXME: currently assumes timestep is one month
        index = self.general_random.filter_for_probability(event.index, t)

        # for those who show up, emit_event that the visit has happened, and tally the cost
        event.population_view.update(pd.Series(event.time, index=index))
        self.general_healthcare_access_emitter(event.split(index))
        self.general_access_count += len(index)

        year = event.time.year
        self.cost_by_year[year] += len(index) * self._appointment_cost(year)
        self.outpatient_cost[year] += len(index) * self._appointment_cost(year)

    @listens_for('time_step')
    @uses_columns(['healthcare_last_visit_date', 'healthcare_followup_date', 'adherence_category'],
                  "alive == 'alive'")
    def followup_access(self, event):
        # determine population due for a follow-up appointment
        rows = ((event.population.healthcare_followup_date > self.clock())
                & (event.population.healthcare_followup_date <= event.time))
        affected_population = event.population[rows]

        # of them, determine who shows up for their follow-up appointment
        adherence = pd.Series(1, index=affected_population.index)
        adherence[affected_population.adherence_category == 'non-adherent'] = 0
        semi_adherents = affected_population.loc[affected_population.adherence_category == 'semi-adherent']
        adherence[semi_adherents.index] = self.semi_adherent_pr
        affected_population = self.followup_random.filter_for_probability(affected_population, adherence)

        # for those who show up, emit_event that the visit has happened, and tally the cost
        event.population_view.update(pd.Series(event.time, index=affected_population.index,
                                               name='healthcare_last_visit_date'))
        self.followup_healthcare_access_emitter(event.split(affected_population.index))
        self.followup_access_count += len(affected_population)

        year = event.time.year
        self.cost_by_year[year] += len(affected_population) * self._appointment_cost(year)
        self.outpatient_cost[year] += len(affected_population) * self._appointment_cost(year)

    @listens_for('hospitalization')
    def hospital_access(self, event):
        year = event.time.year
        self.hospitalization_count += len(event.index)
        self.hospitalization_cost[year] += len(event.index) * self._hospitalization_cost(year)
        self.cost_by_year[year] += len(event.index) * self._hospitalization_cost(year)

    @modifies_value('metrics')
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
