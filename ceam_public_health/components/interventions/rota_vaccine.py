import pandas as pd
import numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config
from ceam.framework.randomness import choice
import pdb


def _determine_who_should_receive_dose(population, vaccine_col, true_weight, dose_age, dose_number):
    """
    Uses choice to determine if simulant should receive a dose. Returns a population of simulants that should receive a dose (most of the time this will be an empty population

    Parameters
    ----------
    population: pd.DataFrame
        population view of all of the simulants who are currently alive
    
    vaccine_col: str
        str representing the name of a column, either rotaviral_entiritis_vaccine_first_dose or second dose. The column represents whether or not the simulant received the first and second dose of the vaccine, which is important because we want to make sure that only people who got the previous vaccine can get the vaccine that is currently being modelled

    true_weight: float
        number between 0 and 1 that represents the probability of being vaccinated

    dose_age: number
        age in days at which simulant should receive specific dose

    dose_number: int
        1, 2, or 3 depending on which dose is currently being evaluated

    Used by
    -------
    determine_who_should_receive_dose
    """

    false_weight = 1 - true_weight
    previous_dose = dose_number - 1

    if previous_dose == 0:
        children_at_dose_age = population.query("age_in_days == @dose_age").copy()

    elif previous_dose == 1:
        children_at_dose_age = population.query("age_in_days == @dose_age and rotaviral_entiritis_vaccine_first_dose == 1").copy()

    elif previous_dose == 2:
        children_at_dose_age = population.query("age_in_days == @dose_age and rotaviral_entiritis_vaccine_second_dose == 1").copy()

    else:
         raise ValueError, "previous_dose cannot be any value other than 1 or 2 or None"

    if not children_at_dose_age.empty:
        children_at_dose_age[vaccine_col] = choice('determine_who_should_receive_dose_{}'.format(vaccine_dose_number), children_at_dose_age.index, [1, 0], [true_weight, false_weight])

    children_who_will_receive_dose = children_at_dose_age.query("{} == 1".format(vaccine_col))
    return children_who_will_receive_dose
      

def determine_who_should_receive_dose(population, index, vaccine_col, dose_number):
    """
    Function will determine who should receive 1st, 2nd, and 3rd doses of a vaccine based on proportions and when they should receive each dose based on info specified in the config file

    Parameters
    ----------
    population: df
        population view of all of the simulants who are currently alive

    index: pandas index 
        index is just the index of all simulants who are currently alive.        

    vaccine_col: str
        str representing the name of a column, either rotaviral_entiritis_vaccine_first_dose or second dose. The column represents whether or not the simulant received the first and second dose of the vaccine, which is important because we want to make sure that only people who got the previous vaccine can get the vaccine that is currently being modelled

    dose_number: int
        1, 2, or 3 depending on which dose is currently being evaluated
    """

    population['age_in_days'] = population['age'] * 365

    population['age_in_days'] = population['age_in_days'].astype(int)

    if dose_number == 1:
        true_weight = config.getfloat('rota_vaccine', 'vaccination_proportion_increase')
        dose_age = config.getint('rota_vaccine', 'age_at_first_dose')

    if dose_number == 2:
        true_weight = config.getint('rota_vaccine', 'second_dose_retention')
        dose_age = config.getint('rota_vaccine', 'age_at_first_dose') + 61

    if dose_number == 3:
        true_weight = config.getint('rota_vaccine', 'third_dose_retention')
        dose_age = config.getint('rota_vaccine', 'age_at_first_dose') + 61 + 61

    # TODO: Make the proportion to vaccinate include the baseline vaccination rates
    children_who_will_receive_dose = _determine_who_should_receive_dose(population=population, vaccine_col=vaccine_col, true_weight=true_weight, dose_age=dose_age, dose_number=dose_number)
       
        return children_who_will_receive_dose

    # If nobody is at the age at which they need to start being vaccinated, just return the original input population so that nothing in the population table changes in the ensuing population update
    return population


class RotaVaccine():
    """
    Class that determines who gets vaccinated, how the vaccine affects incidence, and counts vaccinations
    """

    def __init__(self):
        self.active = config.getboolean('rota_vaccine', 'run_intervention')
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

        self.vaccine_duration_start_time =  self.etiology + "_vaccine_duration_start_time"
        self.vaccine_duration_end_time = self.etiology + "_vaccine_duration_end_time"
        self.vaccine_working_column = self.etiology + "_vaccine_is_working"

        self.vaccine_unit_cost_column = self.etiology + "_vaccine_unit_cost"
        self.vaccine_cost_to_administer_column = "cost_to_administer_" + self.etiology + "_vaccine"

    def setup(self, builder):

        columns = [self.vaccine_first_dose_column, self.vaccine_second_dose_column, self.vaccine_third_dose_column, 
                   self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column, self.vaccine_first_dose_time_column, 
                   self.vaccine_second_dose_time_column, self.vaccine_third_dose_time_column, self.vaccine_first_dose_count_column,
                   self.vaccine_second_dose_count_column, self.vaccine_third_dose_count_column, 
                   self.vaccine_duration_start_time, self.vaccine_duration_end_time, self.vaccine_working_column, 'age']

        self.population_view = builder.population_view(columns, query='alive')


    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({self.vaccine_first_dose_column: np.zeros(len(event.index), dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_second_dose_column: np.zeros(len(event.index), dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_third_dose_column: np.zeros(len(event.index), dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({self.vaccine_first_dose_count_column: np.zeros(len(event.index), dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_second_dose_count_column: np.zeros(len(event.index), dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_third_dose_count_column: np.zeros(len(event.index), dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({self.vaccine_first_dose_time_column: [pd.NaT]*len(event.index)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_second_dose_time_column: [pd.NaT]*len(event.index)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_third_dose_time_column: [pd.NaT]*len(event.index)}, index=event.index))

        self.population_view.update(pd.DataFrame({self.vaccine_duration_start_time: [pd.NaT]*len(event.index)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_duration_end_time: [pd.NaT]*len(event.index)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_working_column: np.zeros(len(event.index), dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({self.vaccine_unit_cost_column: np.zeros(len(event.index), dtype=float)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.vaccine_cost_to_administer_column: np.zeros(len(event.index), dtype=int)}, index=event.index))


    # FIXME: An emitter could potentially be faster. Could have an emitter that says when people reach a certain age, give them a vaccine dose.
    @listens_for('time_step')
    @uses_columns(['age', 'rotaviral_entiritis_vaccine_first_dose', 'rotaviral_entiritis_vaccine_second_dose', 'rotaviral_entiritis_vaccine_third_dose', 'rotaviral_entiritis_vaccine_first_dose_count', 'rotaviral_entiritis_vaccine_second_dose_count', 'rotaviral_entiritis_vaccine_third_dose_count', 'rotaviral_entiritis_vaccine_first_dose_event_time', 'rotaviral_entiritis_vaccine_second_dose_event_time', 'rotaviral_entiritis_vaccine_third_dose_event_time', 'rotaviral_entiritis_vaccine_unit_cost', 'cost_to_administer_rotaviral_entiritis_vaccine', 'rotaviral_entiritis_vaccine_duration_start_time', 'rotaviral_entiritis_vaccine_duration_end_time', 'rotaviral_entiritis_vaccine_is_working'], 'alive')
    def _determine_who_gets_vaccinated(self, event):
        """
        Each time step, call the _determine_who_should_receive_dose function to see which patients should get dosed. Everytime step, do this for all 3 vaccines.
        """
        if self.active != True:
            return

        else:
            population = self.population_view.get(event.index)

            children_who_will_receive_first_dose = determine_who_should_receive_dose(population, event.index, self.vaccine_first_dose_column, 1)

            if not children_who_will_receive_first_dose.empty:
                # Setting time here in case we want to use an emitter in the future
                children_who_will_receive_first_dose[self.vaccine_first_dose_time_column] = pd.Timestamp(event.time)

                # Count vaccination dose
                children_who_will_receive_first_dose[self.vaccine_first_dose_count_column] += 1 

                # Accrue cost
                children_who_will_receive_first_dose[self.vaccine_unit_cost_column] += config.getfloat('rota_vaccine', 'RV5_dose_cost')
                children_who_will_receive_first_dose[self.vaccine_cost_to_administer_column] += config.getint('rota_vaccine', 'cost_to_administer_each_dose')

                self.population_view.update(children_who_will_receive_first_dose[[self.vaccine_first_dose_column, self.vaccine_first_dose_time_column, self.vaccine_first_dose_count_column, self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column]])

            # Second dose
            children_who_will_receive_second_dose = determine_who_should_receive_dose(population, event.index, self.vaccine_second_dose_column, 2)

            if not children_who_will_receive_second_dose.empty:
                # Setting time here in case we want to use an emitter in the future
                children_who_will_receive_second_dose[self.vaccine_second_dose_time_column] = pd.Timestamp(event.time)

                # Count vaccination dose
                children_who_will_receive_second_dose[self.vaccine_second_dose_count_column] += 1

                # Accrue cost
                children_who_will_receive_second_dose[self.vaccine_unit_cost_column] += config.getfloat('rota_vaccine', 'RV5_dose_cost')
                children_who_will_receive_second_dose[self.vaccine_cost_to_administer_column] += config.getint('rota_vaccine', 'cost_to_administer_each_dose')

                self.population_view.update(children_who_will_receive_second_dose[[self.vaccine_second_dose_column, self.vaccine_second_dose_time_column, self.vaccine_second_dose_count_column, self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column]])


            # Third dose
            children_who_will_receive_third_dose = determine_who_should_receive_dose(population, event.index, self.vaccine_third_dose_column, 3)

            if not children_who_will_receive_third_dose.empty:
                # Setting time here in case we want to use an emitter in the future
                children_who_will_receive_third_dose[self.vaccine_third_dose_time_column] = pd.Timestamp(event.time)

                # Count vaccination dose
                children_who_will_receive_third_dose[self.vaccine_third_dose_count_column] += 1

                # Accrue cost
                children_who_will_receive_third_dose[self.vaccine_unit_cost_column] += config.getfloat('rota_vaccine', 'RV5_dose_cost')
                children_who_will_receive_third_dose[self.vaccine_cost_to_administer_column] += config.getint('rota_vaccine', 'cost_to_administer_each_dose')

                # set time at which immunity starts
                time_after_dose_at_which_immunity_is_conferred = config.getint('rota_vaccine', 'time_after_dose_at_which_immunity_is_conferred')
                children_who_will_receive_third_dose[self.vaccine_duration_start_time] = children_who_will_receive_third_dose[self.vaccine_third_dose_time_column] + pd.to_timedelta(time_after_dose_at_which_immunity_is_conferred, unit='D')

                # set time for which immunity will last
                vaccine_duration = config.getint('rota_vaccine', 'vaccine_duration')
                children_who_will_receive_third_dose[self.vaccine_duration_end_time] = children_who_will_receive_third_dose[self.vaccine_duration_start_time] + pd.to_timedelta(vaccine_duration, unit='D')

                self.population_view.update(children_who_will_receive_third_dose[[self.vaccine_third_dose_column, self.vaccine_third_dose_time_column, self.vaccine_third_dose_count_column, self.vaccine_unit_cost_column, self.vaccine_cost_to_administer_column, self.vaccine_duration_start_time, self.vaccine_duration_end_time]])

            current_time = pd.Timestamp(event.time)

            # set flags for when someone should be immunized or not
            population.loc[current_time >= population[self.vaccine_duration_start_time], self.vaccine_working_column] = 1
            population.loc[current_time >= population[self.vaccine_duration_end_time], self.vaccine_working_column] = 0

            self.population_view.update(population[[self.vaccine_working_column]])


    @modifies_value('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    @uses_columns(['diarrhea_due_to_rotaviral_entiritis', 'rotaviral_entiritis_vaccine_third_dose', 'rotaviral_entiritis_vaccine_is_working'], 'alive')
    def incidence_rates(self, index, rates, population_view):
        """
        If the intervention is running, determine who is currently receiving the intervention and then decrease their incidence of diarrhea due to rota by the effectiveness specified in the config file

        Parameters
        ----------
        index: pandas index
            index of all simulants

        rates: pd.Series
            incidence rates for diarrhea due to rotavirus

        population_view: pd.DataFrame
            dataframe of all simulants that are alive with columns diarrhea_due_to_rotaviral_entiritis, rotaviral_entiritis_vaccine_third_dose, rotaviral_entiritis_vaccine_is_working
        """
        population = self.population_view.get(index)

        vaccine_effectiveness = config.getfloat('rota_vaccine', 'total_vaccine_effectiveness')

        if self.active == True:

            if population.query("{} == 1".format(self.vaccine_working_column)).empty:

                return rates

            else:
                # filter population to only people for whom the vaccine is working
                pop = population.query("{} == 1".format(self.vaccine_working_column)).copy()
 
                rates.loc[pop.index] *= (1 - vaccine_effectiveness)

                return rates
        else:
            return rates


    @modifies_value('metrics')
    @uses_columns(['rotaviral_entiritis_vaccine_first_dose_count', 'rotaviral_entiritis_vaccine_second_dose_count', 'rotaviral_entiritis_vaccine_third_dose_count', 'rotaviral_entiritis_vaccine_unit_cost', 'cost_to_administer_rotaviral_entiritis_vaccine'])
    def metrics(self, index, metrics, population_view):
        """
        Update the output metrics with information regarding the vaccine intervention

        Parameters
        ----------
        index: pandas Index
            Index of all simulants, alive or dead

        metrics: pd.Dictionary
            Dictionary of metrics that will be printed out at the end of the simulation

        population_view: pd.DataFrame
            df of all simulants, alive or dead with columns rotaviral_entiritis_vaccine_first_dose_count, rotaviral_entiritis_vaccine_second_dose_count, rotaviral_entiritis_vaccine_third_dose_count, rotaviral_entiritis_vaccine_unit_cost, cost_to_administer_rotaviral_entiritis_vaccine
        """

        population = population_view.get(index)

        metrics['rotaviral_entiritis_vaccine_first_dose_count'] = population['rotaviral_entiritis_vaccine_first_dose_count'].sum()
        metrics['rotaviral_entiritis_vaccine_second_dose_count'] = population['rotaviral_entiritis_vaccine_second_dose_count'].sum()
        metrics['rotaviral_entiritis_vaccine_third_dose_count'] = population['rotaviral_entiritis_vaccine_third_dose_count'].sum()

        metrics[self.vaccine_unit_cost_column] = population[self.vaccine_unit_cost_column].sum()
        metrics[self.vaccine_cost_to_administer_column] = population[self.vaccine_cost_to_administer_column].sum()

        return metrics


# End. 
