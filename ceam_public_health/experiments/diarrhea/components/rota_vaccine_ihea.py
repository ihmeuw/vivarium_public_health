import pandas as pd
import numpy as np

from datetime import datetime

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value

from ceam_inputs import get_rota_vaccine_coverage, get_dtp3_coverage, get_rota_vaccine_protection


class RotaVaccine:

    configuration_defaults = {
        'rota_vaccine': {
            'RV5_dose_cost': 3.5,
            'cost_to_administer_each_dose': 0,
            'vaccine_full_immunity_duration': 730,
            'waning_immunity_time': 0,
            'age_at_first_dose': 61,
            'age_at_second_dose': 122,
            'age_at_third_dose': 183,
            'second_dose_retention': 1,
            'third_dose_retention': 1,
            'vaccination_proportion_increase': 0,
            'time_after_dose_at_which_immunity_is_conferred': 14,
            'dtp3_coverage': 0,
        }
    }

    def __init__(self):
        self.name = 'rotaviral_entiritis_vaccine'
        self.event_time_column = "{}_event_time".format(self.name)

        self.immunity_delay = pd.to_timedelta(
            config.rota_vaccine.time_after_dose_at_which_immunity_is_conferred, unit='D')
        self.immunity_duration = pd.to_timedelta(config.rota_vaccine.vaccine_full_immunity_duration, unit='D')
        self.waning_time = pd.to_timedelta(config.rota_vaccine.waning_immunity_time, unit='D')

        self.doses = ['first', 'second', 'third']
        self.dose_ages = {dose: config.rota_vaccine['age_at_{}_dose'.format(dose)]/365 for dose in self.doses}
        self.retention = {'first': 1,
                          'second': config.rota_vaccine.second_dose_retention,
                          'third': config.rota_vaccine.third_dose_retention}
        self._coverage_data = get_dtp3_coverage() if config.rota_vaccine.dtp3_coverage else get_rota_vaccine_coverage()

    def setup(self, builder):
        self.clock = builder.clock()
        self.population_view = builder.population_view([self.name, self.event_time_column, 'age'],
                                                       query="alive == 'alive'")
        self.randomness = {dose: builder.randomness('{}_dose_randomness'.format(dose)) for dose in self.doses}

        self.coverage = builder.value('{}_coverage'.format(self.name))
        self.coverage.source = self._coverage_data
        self.protection = {'none': 0,
                           'first': config.rota_vaccine.first_dose_protection,
                           'second': config.rota_vaccine.second_dose_protection,
                           'third': get_rota_vaccine_protection()}

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(
            pd.DataFrame({self.name: pd.Series(0, index=event.index),
                          "{}_event_time".format(self.name): pd.Series(pd.NaT, index=event.index)})
        )

    @listens_for('time_step')
    def administer_vaccine(self, event):
        population = self.population_view.get(event.index)
        for n, dose in enumerate(self.doses):
            dosed_population = population.iloc[self.determine_who_should_receive_dose(population, n)]
            dosed_population[self.name] += 1
            dosed_population[self.event_time_column] = self.clock()
            self.population_view.update(dosed_population)

    def determine_who_should_receive_dose(self, population, dose_number):
        dose = self.doses[dose_number]
        previous_dose = dose_number - 1

        dt = config.simulation_parameters.time_step
        correct_age = np.abs(population.age - self.dose_ages[dose]) < dt/365
        got_previous_dose = population[self.name] == previous_dose
        eligible_children_index = population[correct_age & got_previous_dose].index

        # FIXME: GBD coverage metric is a measure of people that receive all 3
        # vaccines, not just 1. Need to figure out a way to capture this in the model
        if dose_number == 1:
            # Vaccine was approved on February 3, 2006. Need lines below to account for dtp3.
            # Want to say our intervention starts first day it's theoretically possible
            coverage = self.coverage(eligible_children_index) if self.clock() > datetime(2006, 2, 4, 0, 0) else 0
        else:
            coverage = self.retention[dose]

        return self.randomness[dose].filter_for_probability(eligible_children_index, coverage)

    @modifies_value('incidence_rate.rotaviral_entiritis')
    def incidence_rates(self, index, rates):
        population = self.population_view.get(index)
        protection = self.determine_vaccine_protection(population)

        rates.loc[index] *= (1 - protection)

        return rates

    def determine_vaccine_protection(self, pop):
        immunity_start = pop[self.event_time_column] + self.immunity_delay
        full_immunity_end = immunity_start + self.immunity_duration
        waning_immunity_end = full_immunity_end + self.waning_time

        full_immunity = ((self.clock() >= immunity_start) & (self.clock() < full_immunity_end))
        waning_immunity = ((self.clock() >= full_immunity_end) & (self.clock() < waning_immunity_end))

        time_in_waning_protection = waning_immunity_end[waning_immunity] - full_immunity_end[waning_immunity]

        protection = pd.Series(0, index=pop.index)
        for n, dose in enumerate(self.doses):
            protection[full_immunity & (pop[self.name] == dose)] = self.protection[dose]
            protection_slope = self.protection[dose]/self.waning_time
            protection[waning_immunity & (pop[self.name] == dose)] = (self.protection[dose]
                                                                      - protection_slope*time_in_waning_protection)

        return protection

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        count_vacs = population.groupby(self.name).size()

        metrics['rotaviral_entiritis_vaccine_first_dose_count'] = count_vacs[-3:].sum()
        metrics['rotaviral_entiritis_vaccine_second_dose_count'] = count_vacs[-2:].sum()
        metrics['rotaviral_entiritis_vaccine_third_dose_count'] = count_vacs[-1:].sum()

        total_number_of_administered_vaccines = metrics['rotaviral_entiritis_vaccine_first_dose_count'] + \
                                                metrics['rotaviral_entiritis_vaccine_second_dose_count'] + \
                                                metrics['rotaviral_entiritis_vaccine_third_dose_count']

        metrics['vaccine_unit_cost_column'] = total_number_of_administered_vaccines * config.rota_vaccine.RV5_dose_cost
        metrics['vaccine_cost_to_administer_column'] = (total_number_of_administered_vaccines
                                                        * config.rota_vaccine.cost_to_administer_each_dose)

        return metrics
