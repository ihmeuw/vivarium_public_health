from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value

import ceam_public_health.components.healthcare_access

from ceam_inputs import get_hypertension_drug_costs

# TODO: This feels like configuration but is difficult to express in ini type files.
MEDICATIONS = [
    {
        'name': 'Thiazide-type diuretics',
        'daily_cost': 0.009,
        'efficacy_mean': 8.8,
        'efficacy_sd': .281,
    },
    {
        'name': 'Beta blockers',
        'daily_cost': 0.048,
        'efficacy_mean': 9.2,
        'efficacy_sd': .332,
    },
    {
        'name': 'ACE Inhibitors',
        'daily_cost': 0.059,
        'efficacy_mean': 10.3,
        'efficacy_sd': .281,
    },
    {
        'name': 'Calcium-channel blockers',
        'daily_cost': 0.166,
        'efficacy_mean': 8.8,
        'efficacy_sd': .23,
    },
]


def _hypertensive_categories(population):
    hypertensive_threshold = config.opportunistic_screening.hypertensive_threshold
    severe_hypertensive_threshold = config.opportunistic_screening.severe_hypertensive_threshold
    under_60 = population.age < 60
    over_60 = population.age >= 60
    under_hypertensive = population.high_systolic_blood_pressure_exposure < hypertensive_threshold
    under_hypertensive_older = population.high_systolic_blood_pressure_exposure < hypertensive_threshold+10
    under_severe_hypertensive = population.high_systolic_blood_pressure_exposure < severe_hypertensive_threshold

    normotensive = (under_60 & under_hypertensive) | (over_60 & under_hypertensive_older)
    severe_hypertension = (~under_severe_hypertensive)
    hypertensive = ~(normotensive | severe_hypertension)

    return population.loc[normotensive], population.loc[hypertensive], population.loc[severe_hypertension]


class OpportunisticScreening:
    """Model an intervention where simulants have their blood pressure tested every time they 
    access health care and are prescribed blood pressure reducing medication if they are found 
    to be hypertensive. Each simulant can be prescribed up to `config.opportunistic_screening.max_medications` 
    drugs. If they are still hypertensive while taking all the drugs then there is no further treatment.

    Population Columns
    ------------------
    medication_count : int
    MEDICATION_supplied_until : pd.Timestamp
    """

    configuration_defaults = {
            'opportunistic_screening': {
                'max_medications': 4,
                'blood_pressure_test_cost': 2.43,
                'hypertensive_threshold': 140,
                'severe_hypertensive_threshold': 180,
                'minimum_age_to_screen': 0,
            }
    }

    def __init__(self, active=True):
        self.active = active
        self.cost_by_year = defaultdict(int)

    def setup(self, builder):
        self.cost_by_year = defaultdict(int)

        # draw random costs and effects for medications
        draw = config.run_configuration.draw_number
        r = np.random.RandomState(12345+draw)
        cost_df = get_hypertension_drug_costs()

        for med in MEDICATIONS:
            med['efficacy'] = r.normal(loc=med['efficacy_mean'], scale=med['efficacy_sd'])
            med['daily_cost'] = cost_df.loc[med['name'], 'draw_{}'.format(draw)]

        self.semi_adherent_efficacy = r.normal(0.4, 0.0485)

        assert config.opportunistic_screening.max_medications <= len(MEDICATIONS), 'cannot model more medications than we have data for'

        columns = ['medication_count', 'adherence_category', 'high_systolic_blood_pressure_exposure', 'age', 'healthcare_followup_date', 'healthcare_last_visit_date', 'last_screening_date']

        for medication in MEDICATIONS:
            columns.append(medication['name']+'_supplied_until')
        self.population_view = builder.population_view(columns, query='alive')

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        # TODO: Some people will start out taking medications?
        population = pd.DataFrame({'medication_count': np.zeros(len(event.index), dtype=int),
                                   'last_screening_date': [pd.NaT]*len(event.index)})
        for medication in MEDICATIONS:
            population[medication['name']+'_supplied_until'] = pd.NaT
        self.population_view.update(population)

    def _medication_costs(self, population, current_time):
        # TODO: shouldn't have to dip back into the central table like this
        population = self.population_view.get(population.index)

        for medication_number, medication in enumerate(MEDICATIONS):
            affected_population = population[population.medication_count > medication_number]
            if not affected_population.empty:
                supply_remaining = affected_population[medication['name']+'_supplied_until'] - current_time
                supply_remaining = supply_remaining.fillna(pd.Timedelta(days=0))
                idx = supply_remaining < pd.Timedelta(days=0)
                supply_remaining[idx] = pd.Series([pd.Timedelta(days=0)]*idx.sum())

                supply_needed = affected_population['healthcare_followup_date'] - current_time
                supply_needed = supply_needed.fillna(pd.Timedelta(days=0))
                supply_needed[supply_needed < pd.Timedelta(days=0)] = pd.Timedelta(days=0)

                supplied_until = current_time + pd.DataFrame([supply_needed, supply_remaining]).T.max(axis=1)
                if self.active:
                    self.population_view.update(pd.Series(supplied_until,
                                                          index=affected_population.index,
                                                          name=medication['name']+'_supplied_until'))
                annual_cost = max(0, (supply_needed - supply_remaining).dt.days.sum()) * medication['daily_cost']
                self.cost_by_year[current_time.year] += annual_cost

    @listens_for('general_healthcare_access')
    def general_blood_pressure_test(self, event):
        #TODO: Model blood pressure testing error
        if self.active:

            minimum_age_to_screen = config.opportunistic_screening.minimum_age_to_screen
            affected_population = self.population_view.get(event.index)
            affected_population = affected_population[affected_population.age >= minimum_age_to_screen]

            year = event.time.year
            appointment_cost = ceam_public_health.components.healthcare_access.appointment_cost[year]
            cost_per_simulant = appointment_cost * 0.25  # see CE-94 for discussion
            self.cost_by_year[year] += cost_per_simulant * len(affected_population)

            normotensive, hypertensive, severe_hypertension = _hypertensive_categories(affected_population)

            # Normotensive simulants get a 60 month followup and no drugs
            self.population_view.update(pd.Series(event.time + timedelta(days=30.5*60),
                                                  index=normotensive.index, name='healthcare_followup_date'))

            # Hypertensive simulants get a 1 month followup and no drugs
            self.population_view.update(pd.Series(event.time + timedelta(days=30.5),
                                                  index=hypertensive.index, name='healthcare_followup_date'))

            # Severe hypertensive simulants get a 1 month followup and two drugs
            self.population_view.update(pd.Series(event.time + timedelta(days=30.5*6),
                                                  index=severe_hypertension.index, name='healthcare_followup_date'))

            self.population_view.update(pd.Series(np.minimum(severe_hypertension['medication_count'] + 2,
                                                             config.opportunistic_screening.max_medications),
                                                  name='medication_count'))

            self._medication_costs(affected_population, event.time)

            self.population_view.update(pd.Series(event.time,
                                                  index=affected_population.index, name='last_screening_date'))


    @listens_for('followup_healthcare_access')
    def followup_blood_pressure_test(self, event):
        if self.active:

            year = event.time.year
            appointment_cost = ceam_public_health.components.healthcare_access.appointment_cost[year]
            cost_per_simulant = appointment_cost

            affected_population = self.population_view.get(event.index)
            self.cost_by_year[year] += cost_per_simulant * len(affected_population)
            normotensive, hypertensive, severe_hypertension = _hypertensive_categories(affected_population)

            nonmedicated_normotensive = normotensive.loc[normotensive.medication_count == 0]
            medicated_normotensive = normotensive.loc[normotensive.medication_count > 0]

            # Unmedicated normotensive simulants get a 60 month followup
            follow_up = event.time + timedelta(days=30.5*60)
            self.population_view.update(pd.Series(follow_up,
                                                  index=nonmedicated_normotensive.index,
                                                  name='healthcare_followup_date'))

            # Medicated normotensive simulants get an 11 month followup
            follow_up = event.time + timedelta(days=30.5*11)
            self.population_view.update(pd.Series(follow_up,
                                                  index=medicated_normotensive.index, name='healthcare_followup_date'))

            # Hypertensive simulants get a 6 month followup and go on one drug
            follow_up = event.time + timedelta(days=30.5*6)
            self.population_view.update(pd.Series(follow_up,
                                                  index=hypertensive.index.append(severe_hypertension.index),
                                                  name='healthcare_followup_date'))
            self.population_view.update(pd.Series(np.minimum(hypertensive['medication_count'] + 1,
                                                             config.opportunistic_screening.max_medications),
                                                  index=hypertensive.index, name='medication_count'))
            self.population_view.update(pd.Series(np.minimum(severe_hypertension.medication_count + 1,
                                                             config.opportunistic_screening.max_medications),
                                                  index=severe_hypertension.index, name='medication_count'))

            self._medication_costs(affected_population, event.time)

    @listens_for('time_step__prepare', priority=9)
    def adjust_blood_pressure(self, event):
        if self.active:
            time_step = timedelta(days=config.simulation_parameters.time_step)
            for medication_number, medication in enumerate(MEDICATIONS):
                initial_affected_population = self.population_view.get(event.index)
                affected_population = initial_affected_population[
                    (initial_affected_population.medication_count > medication_number)
                    & (initial_affected_population[medication['name']+'_supplied_until'] >= event.time - time_step)]
                adherence = pd.Series(1, index=affected_population.index)
                adherence[affected_population.adherence_category == 'non-adherent'] = 0
                semi_adherents = affected_population.loc[affected_population.adherence_category == 'semi-adherent']
                adherence[semi_adherents.index] = self.semi_adherent_efficacy

                medication_efficacy = medication['efficacy'] * adherence
                affected_population = affected_population.copy()
                affected_population['high_systolic_blood_pressure_exposure'] -= medication_efficacy
                self.population_view.update(affected_population['high_systolic_blood_pressure_exposure'])

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        metrics['medication_cost'] = sum(self.cost_by_year.values())
        if 'cost' in metrics:
            metrics['cost'] += metrics['medication_cost']
        else:
            metrics['cost'] = metrics['medication_cost']
        pop = self.population_view.get(index)
        metrics['treated_individuals'] = (pop.medication_count > 0).sum()
        metrics['screened_simulants'] = (~pop.last_screening_date.isnull()).sum()
        return metrics
