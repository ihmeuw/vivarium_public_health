import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config
from ceam.framework.randomness import choice


# 2 things that really need to happen in this code -- set up emitters to do follow up vaccinations and fix how I'm grabbing the age columni. Figure out how to confer immunity 1 month after 3rd dose received.


def determine_who_should_receive_dose(population, index, vaccine_col, dose_number):

    population['age_in_days'] = population['age'] * 365

    population['age_in_days'] = population['age_in_days'].astype(int)

    if dose_number == 1:
        vaccination_proportion_increase = config.getfloat('rota_vaccine', 'vaccination_proportion_increase')
        # TODO: Make the proportion to vaccinate include the baseline vaccination rates
        true_weight = vaccination_proportion_increase
        false_weight = 1 - true_weight
        dose_age = config.getfloat('rota_vaccine', 'age_at_first_dose')
        children_at_dose_age = population.query("age_in_days == @dose_age").copy()
       
        if not children_at_dose_age.empty:
            children_at_dose_age[vaccine_col] = choice('determine_who_should_receive_first_dose', index, ['True', 'False'], [true_weight, false_weight])

        children_who_will_receive_dose = children_at_dose_age.query("{} == True".format(vaccine_col))

        return children_who_will_receive_dose[[vaccine_col]]

    if dose_number == 2:
        second_dose_retention = config.getint('rota_vaccine', 'second_dose_retention')
        true_weight = second_dose_retention
        false_weight = 1 - true_weight
        # give second dose 2 months after first
        dose_age = config.getfloat('rota_vaccine', 'age_at_first_dose') + 61
        children_at_dose_age = population.query("age_in_days == @dose_age and rotaviral_entiritis_vaccine_first_dose == True").copy()
       
        if not children_at_dose_age.empty:
            children_at_dose_age[vaccine_col] = choice('determine_who_should_receive_second_dose', index, ['True', 'False'], [true_weight, false_weight])

        children_who_will_receive_dose = children_at_dose_age.query("{} == True".format(vaccine_col))

        return children_who_will_receive_dose[[vaccine_col]]

    if dose_number == 3:
        third_dose_retention = config.getint('rota_vaccine', 'third_dose_retention')
        true_weight = third_dose_retention
        false_weight = 1 - true_weight
        # give third dose 4 months after first dose
        dose_age = config.getfloat('rota_vaccine', 'age_at_first_dose') + 61 + 61
        children_at_dose_age = population.query("age_in_days == @dose_age and rotaviral_entiritis_vaccine_second_dose == True").copy()

        if not children_at_dose_age.empty:
            children_at_dose_age[vaccine_col] = choice('determine_who_should_receive_third_dose', index, ['True', 'False'], [true_weight, false_weight])

        children_who_will_receive_dose = children_at_dose_age.query("{} == True".format(vaccine_col))

        return children_who_will_receive_dose[[vaccine_col]]
   
    # If nobody is at the age at which they need to start being vaccinated, just return the original input population so that nothing in the population table changes in the ensuing population update
    return population

class RotaVaccine():
    def __init__(self, active=True):
        self.active = active
        self.etiology = 'rotaviral_entiritis'
        self.etiology_column = 'diarrhea_due_to_' + self.etiology

        self.vaccine_first_dose_column = self.etiology + "_vaccine_first_dose"
        self.vaccine_second_dose_column = self.etiology + "_vaccine_second_dose"
        self.vaccine_third_dose_column = self.etiology + "_vaccine_third_dose"

        self.vaccine_first_dose_count_column = self.etiology + "_vaccine_first_dose_count"
        self.vaccine_second_dose_count_column = self.etiology + "_vaccine_second_dose_count"
        self.vaccine_third_dose_count_column = self.etiology + "_vaccine_third_dose_count"

        self.vaccine_first_dose_time_column = self.etiology + "_vaccine_first_dose_event_time"
        self.vaccine_second_dose_time_column = self.etiology + "_vaccine_second_dose_event_time"
        self.vaccine_third_dose_time_column = self.etiology + "_vaccine_third_dose_event_time"

        self.vaccine_unit_cost_column = self.etiology + "_vaccine_unit_cost"
        self.vaccine_cost_to_administer_column = "cost_to_administer_" + self.etiology + "_vaccine"

    def setup(self, builder):
        self.vaccine_cost_by_year = 0
        self.vaccine_administration_cost_by_year = 0

        columns = [self.vaccine_first_dose_column, self.vaccine_second_dose_column, self.vaccine_third_dose_column, 
                   self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column, self.vaccine_first_dose_time_column, 
                   self.vaccine_second_dose_time_column, self.vaccine_third_dose_time_column, self.vaccine_first_dose_count_column,
                   self.vaccine_second_dose_count_column, self.vaccine_third_dose_count_column, 'age']

        self.population_view = builder.population_view(columns, query='alive')


    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({self.vaccine_first_dose_column: np.zeros(len(event.index), dtype=int)}))
        self.population_view.update(pd.DataFrame({self.vaccine_second_dose_column: np.zeros(len(event.index), dtype=int)}))
        self.population_view.update(pd.DataFrame({self.vaccine_third_dose_column: np.zeros(len(event.index), dtype=int)}))

        self.population_view.update(pd.DataFrame({self.vaccine_first_dose_count_column: np.zeros(len(event.index), dtype=int)}))
        self.population_view.update(pd.DataFrame({self.vaccine_second_dose_count_column: np.zeros(len(event.index), dtype=int)}))
        self.population_view.update(pd.DataFrame({self.vaccine_third_dose_count_column: np.zeros(len(event.index), dtype=int)}))

        self.population_view.update(pd.DataFrame({self.vaccine_first_dose_time_column: [pd.NaT]*len(event.index)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_second_dose_time_column: [pd.NaT]*len(event.index)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_third_dose_time_column: [pd.NaT]*len(event.index)}, index=event.index))

        self.population_view.update(pd.DataFrame({self.vaccine_unit_cost_column: np.zeros(len(event.index), dtype=int)}))
        self.population_view.update(pd.DataFrame({self.vaccine_cost_to_administer_column: np.zeros(len(event.index), dtype=int)}))


    @listens_for('time_step')
    @uses_columns(['age', 'rotaviral_entiritis_vaccine_first_dose', 'rotaviral_entiritis_vaccine_second_dose', 'rotaviral_entiritis_vaccine_third_dose', 'rotaviral_entiritis_vaccine_first_dose_count', 'rotaviral_entiritis_vaccine_second_dose_count', 'rotaviral_entiritis_vaccine_third_dose_count', 'rotaviral_entiritis_vaccine_first_dose_event_time', 'rotaviral_entiritis_vaccine_second_dose_event_time', 'rotaviral_entiritis_vaccine_third_dose_event_time', 'rotaviral_entiritis_vaccine_unit_cost', 'cost_to_administer_rotaviral_entiritis_vaccine'], 'alive')
    def _determine_who_gets_first_dose(self, event):
        population = self.population_view.get(event.index)

        children_who_will_receive_first_dose = determine_who_should_receive_dose(population, event.index, self.vaccine_first_dose_column, 1)

        if not children_who_will_receive_first_dose.empty:
            # Setting time here in case we want to use an emitter in the future
            children_who_will_receive_first_dose[self.vaccine_first_dose_time_column] = event.time

            # Count vaccination dose
            children_who_will_receive_first_dose[self.vaccine_first_dose_count_column] += 1 

            # Accrue cost
            children_who_will_receive_first_dose[self.vaccine_unit_cost_column] += config.getint('rota_vaccine', 'RV5_dose_cost')
            children_who_will_receive_first_dose[self.vaccine_cost_to_administer_column] += config.getint('rota_vaccine', 'cost_to_administer_each_dose')
 
            self.population_view.update(children_who_will_receive_first_dose[[self.vaccine_first_dose_column, self.vaccine_first_dose_time_column, self.vaccine_first_dose_count_column, self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column]])

        # Second dose
        children_who_will_receive_second_dose = determine_who_should_receive_dose(population, event.index, self.vaccine_second_dose_column, 2)

        if not children_who_will_receive_second_dose.empty:
            # Setting time here in case we want to use an emitter in the future
            children_who_will_receive_second_dose[self.vaccine_second_dose_time_column] = event.time

            # Count vaccination dose
            children_who_will_receive_second_dose[self.vaccine_second_dose_count_column] += 1

            # Accrue cost
            children_who_will_receive_second_dose[self.vaccine_unit_cost_column] += config.getint('rota_vaccine', 'RV5_dose_cost')
            children_who_will_receive_second_dose[self.vaccine_cost_to_administer_column] += config.getint('rota_vaccine', 'cost_to_administer_each_dose')

            self.population_view.update(children_who_will_receive_second_dose[[self.vaccine_second_dose_column, self.vaccine_second_dose_time_column, self.vaccine_second_dose_count_column, self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column]])


        # 3rd dose
        children_who_will_receive_third_dose = determine_who_should_receive_dose(population, event.index, self.vaccine_third_dose_column, 3)

        if not children_who_will_receive_third_dose.empty:

            # Setting time here in case we want to use an emitter in the future
            children_who_will_receive_third_dose[self.vaccine_third_dose_time_column] = event.time

            # Count vaccination dose
            children_who_will_receive_third_dose[self.vaccine_third_dose_count_column] += 1

            # Accrue cost
            children_who_will_receive_third_dose[self.vaccine_unit_cost_column] += config.getint('rota_vaccine', 'RV5_dose_cost')
            children_who_will_receive_third_dose[self.vaccine_cost_to_administer_column] += config.getint('rota_vaccine', 'cost_to_administer_each_dose')

            self.population_view.update(children_who_will_receive_third_dose[[self.vaccine_third_dose_column, self.vaccine_third_dose_time_column, self.vaccine_third_dose_count_column, self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column]])


    @modifies_value('incidence_rate.incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    @uses_columns(['diarrhea_due_to_rotaviral_entiritis', 'rotaviral_entiritis_vaccine_third_dose'], 'alive')
    def incidence_rates(self, index, rates, population_view):
        population = self.population_view.get(index)

        vaccine_effectiveness = config.getfloat('rota_vaccine', 'total_vaccine_effectiveness')

        # FIXME: Get this to return rates for only the affected population
        if self.active == True:
            if population[self.vaccine_third_dose_column].sum() != 0:
                return rates * vaccine_effectiveness * (population[self.vaccine_third_dose_column] == True)

            return rates

        else:
            return rates


    @modifies_value('metrics')
    @uses_columns(['rotaviral_entiritis_vaccine_first_dose_count', 'rotaviral_entiritis_vaccine_second_dose_count', 'rotaviral_entiritis_vaccine_third_dose_count', 'rotaviral_entiritis_vaccine_unit_cost', 'cost_to_administer_rotaviral_entiritis_vaccine'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics['rotaviral_entiritis_vaccine_first_dose_count'] = population['rotaviral_entiritis_vaccine_first_dose_count'].sum()
        metrics['rotaviral_entiritis_vaccine_second_dose_count'] = population['rotaviral_entiritis_vaccine_second_dose_count'].sum()
        metrics['rotaviral_entiritis_vaccine_third_dose_count'] = population['rotaviral_entiritis_vaccine_third_dose_count'].sum()

        metrics[self.vaccine_unit_cost_column] = population[self.vaccine_unit_cost_column].sum()
        metrics[self.vaccine_cost_to_administer_column] = population[self.vaccine_cost_to_administer_column].sum()

        return metrics


# End. 
