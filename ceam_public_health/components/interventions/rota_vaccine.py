import pandas as pd
import numpy as np

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam.framework.randomness import choice


#### TODO: CONFIRM WITH IBRAHIM: SHOULD VACCINE LOSE EFFECT 2 YEARS AFTER ITS ADMINISTERED OR 2 YEARS AFTER IT STARTS TO HAVE AN EFFECT?

def _determine_who_should_receive_dose(population, vaccine_col, true_weight,
                                       dose_age, dose_number):
    """
    Uses choice to determine if simulant should receive a dose. Returns a
        population of simulants that should receive a dose (most of the time
        this function will return an empty population)

    Parameters
    ----------
    population: pd.DataFrame
        population view of all of the simulants who are currently alive

    vaccine_col: str
        str representing the name of a column,
        either rotaviral_entiritis_vaccine_first_dose or second dose.
        The column represents whether or not the simulant received the first
        and second dose of the vaccine, which is important because we want to
        make sure that only people who got the previous vaccine can get the
        vaccine that is currently being modelled

    true_weight: float
        number between 0 and 1 that represents the probability of being
        vaccinated

    dose_age: number
        age in days at which simulant should receive specific dose

    dose_number: int
        1, 2, or 3 depending on which dose is currently being evaluated

    Used by
    -------
    determine_who_should_receive_dose
    """

    false_weight = 1 - true_weight
    # TODO: Don't need previous dose, use dose_number instead
    previous_dose = dose_number - 1

    if previous_dose == 0:
        children_at_dose_age = population.query(
            "age_in_days == @dose_age").copy()

    elif previous_dose == 1:
        children_at_dose_age = population.query(
            "age_in_days == @dose_age and" +
            " rotaviral_entiritis_vaccine_first_dose == 1").copy()

    elif previous_dose == 2:
        children_at_dose_age = population.query(
            "age_in_days == @dose_age and" +
            " rotaviral_entiritis_vaccine_second_dose == 1").copy()

    else:
        raise(ValueError, "previous_dose cannot be any value other than" +
                          " 0, 1, or 2")

    if not children_at_dose_age.empty:
        children_at_dose_age[vaccine_col] = choice(
            'determine_who_should_receive_dose_{}'.format(dose_number),
            children_at_dose_age.index, [1, 0], [true_weight, false_weight])

    children_who_will_receive_dose = children_at_dose_age.query(
        "{} == 1".format(vaccine_col))

    return children_who_will_receive_dose


def accrue_vaccine_cost_and_count(population, vaccine_time_column,
                                  vaccine_count_column,
                                  vaccine_unit_cost_column,
                                  vaccine_cost_to_administer_column,
                                  current_time):
    """
    Takes a population of simulants that have just been vaccinated as an input
    along with several column names and a timestamp of the current time. The
    function determines how much money was spent on the vaccines in the current
    time step, how many vaccines were administered, and when vaccines sets a
    column signifying when the vaccines were administered

    Parameters
    ----------
    population: pd.DataFrame
        population of simulants that have just been vaccinated in the current
        time step

    vaccine_time_column: pd.Series
        column indicating which vaccine was just administered. the value of the
        column will be the time at which the vaccine was administered

    vaccine_count_column: pd.Series
        column indicating which vaccine was just administered. the value of the
        column will be the number of vaccines of the specific dose that
        have been administered

    vaccine_cost_to_administer_column: pd.Series
        column indicating which vaccine was just administered. the value of the
        column will be the total cost of administering each dose of the vaccine

    current_time: pd.Timestamp
        Timestamp of the current time in the simulation. we'll set the
        vaccine_time_column value using this timestamp
    """
    # Setting time here in case we want to use an emitter in the future
    population[vaccine_time_column] = current_time

    # Count vaccination dose
    population[vaccine_count_column] += 1

    # Accrue cost
    population[vaccine_unit_cost_column] += config.getfloat('rota_vaccine',
                                                            'RV5_dose_cost')

    population[vaccine_cost_to_administer_column] += config.getint(
        'rota_vaccine', 'cost_to_administer_each_dose')

    return population


def determine_who_should_receive_dose(population, index, vaccine_col,
                                      dose_number):
    """
    Function will determine who should receive 1st, 2nd, and 3rd doses of a
    vaccine based on proportions/age at first dose as specified in the config
    file

    Parameters
    ----------
    population: df
        population view of all of the simulants who are currently alive

    index: pandas index
        index is just the index of all simulants who are currently alive.

    vaccine_col: str
        str representing the name of a column, either
        rotaviral_entiritis_vaccine_first_dose or second dose. The column
        represents whether or not the simulant received the first and second
        dose of the vaccine, which is important because we want to make sure
        that only people who got the previous vaccine can get the vaccine that
        is currently being modelled

    dose_number: int
        1, 2, or 3 depending on which dose is currently being evaluated
    """

    population['age_in_days'] = population['age'] * 365

    population['age_in_days'] = population['age_in_days'].astype(int)

    # FIXME: Need to figure out how to include baseline vaccine coverage
    #     from GBD in the model
    if dose_number == 1:
        true_weight = config.getfloat('rota_vaccine',
                                      'vaccination_proportion_increase')

        dose_age = config.getint('rota_vaccine', 'age_at_first_dose')

    if dose_number == 2:
        true_weight = config.getint('rota_vaccine', 'second_dose_retention')
        dose_age = config.getint('rota_vaccine', 'age_at_first_dose') + 61

    if dose_number == 3:
        true_weight = config.getint('rota_vaccine', 'third_dose_retention')
        dose_age = config.getint('rota_vaccine', 'age_at_first_dose') + 61 + 61

    # TODO: Make the proportion to vaccinate include the baseline vaccination
    #    rates
    children_who_will_receive_dose = _determine_who_should_receive_dose(
        population=population, vaccine_col=vaccine_col,
        true_weight=true_weight, dose_age=dose_age, dose_number=dose_number)

    return children_who_will_receive_dose


def set_vaccine_duration(population, current_time, etiology, dose):
    """ Function that sets vaccine duration
    
    Parameters
    ----------
    population : pd.DataFrame()
        population_view of simulants that have just been vaccinated
        
    current_time: pd.Timestamp
        current time in the simulation
        
    etiology: str
        specific etiology that is being afftected by the vaccine.
        in the case of rota, our etiology is rotaviral entiritis
    
    dose: str
        can be "first", "second", or "third"
    """
    assert dose in ["first", "second", "third"], "dose can be one of first, second, or third"
    
    # determine when effect of the vaccine should start
    time_after_dose_at_which_immunity_is_conferred = config.getint('rota_vaccine',
                'time_after_{}_dose_at_which_immunity_is_conferred'.format(dose))

    population["{e}_vaccine_{d}_dose_duration_start_time".format(e=etiology, d=dose)] = \
        population["{e}_vaccine_{d}_dose_event_time".format(e=etiology, d=dose)] + \
        pd.to_timedelta(time_after_dose_at_which_immunity_is_conferred, unit='D')
    
    # determine when the effect of the vaccine should end
    vaccine_duration = config.getint('rota_vaccine', 'vaccine_duration')
    
    population["{e}_vaccine_{d}_dose_duration_end_time".format(e=etiology, d=dose)] = \
        population["{e}_vaccine_{d}_dose_duration_start_time".format(e=etiology, d=dose)] + \
        pd.to_timedelta(vaccine_duration, unit='D')
        
    return population


def _set_working_column(population, current_time, etiology):
    """
    Function that sets the "working column", a binary column that indicates whether the vaccine is working (1) or not working (0).
    A vaccine will only be working after it has been administered and if the current time is in between the vaccine start and end
    time and the next vaccine isn't working. If the next vaccine is working, then we want to use the effect of the next vaccine,
    so we don't want the previous vaccine to be working. For instance, if a simulant has received 2 doses of a vaccine, we want
    for the benefit of 2 doses, not one dose, to be conferred to the simulant
    
    Parameters
    ----------
    population : pd.DataFrame()
        population_view of simulants
        
    current_time: pd.Timestamp
        current time in the simulation    
    
    etiology: str
        specific etiology that is being afftected by the vaccine.
        in the case of rota, our etiology is rotaviral entiritis
    """
    # set the dose working cols to 1 if  vaccine_duration_start<=current_time<=vaccine_duration_end
    for dose in ["first", "second", "third"]:
        population.loc[(current_time >= population[
            "{e}_vaccine_{d}_dose_duration_start_time".format(e=etiology, d=dose)]) 
            & (current_time <= population[
            "{e}_vaccine_{d}_dose_duration_end_time".format(e=etiology, d=dose)]),
            "{e}_vaccine_{d}_dose_is_working".format(e=etiology, d=dose)] = 1

    # now make sure that the working col is 1 only for the most recently administered vaccine
    # if third dose has been administered, set the first and second working cols to 0
    population.loc[population["rotaviral_entiritis_vaccine_third_dose_is_working"] == 1, "rotaviral_entiritis_vaccine_first_dose_is_working"] = 0
    population.loc[population["rotaviral_entiritis_vaccine_third_dose_is_working"] == 1, "rotaviral_entiritis_vaccine_second_dose_is_working"] = 0

    # if the second dose has been administered, set the first working col to 0
    population.loc[population["rotaviral_entiritis_vaccine_second_dose_is_working"] == 1, "rotaviral_entiritis_vaccine_first_dose_is_working"] = 0

    return population


class RotaVaccine():
    """
    Class that determines who gets vaccinated, how the vaccine affects
    incidence, and counts vaccinations
    """

    def __init__(self):
        self.active = config.getboolean('rota_vaccine', 'run_intervention')
        self.etiology = 'rotaviral_entiritis'
        self.etiology_column = 'diarrhea_due_to_' + self.etiology

        self.vaccine_first_dose_column = self.etiology + "_vaccine_first_dose"
        self.vaccine_second_dose_column = self.etiology + \
            "_vaccine_second_dose"
        self.vaccine_third_dose_column = self.etiology + "_vaccine_third_dose"

        self.vaccine_first_dose_count_column = self.etiology + \
            "_vaccine_first_dose_count"
        self.vaccine_second_dose_count_column = self.etiology + \
            "_vaccine_second_dose_count"
        self.vaccine_third_dose_count_column = self.etiology + \
            "_vaccine_third_dose_count"

        self.vaccine_first_dose_time_column = self.etiology + \
            "_vaccine_first_dose_event_time"
        self.vaccine_second_dose_time_column = self.etiology + \
            "_vaccine_second_dose_event_time"
        self.vaccine_third_dose_time_column = self.etiology + \
            "_vaccine_third_dose_event_time"

        self.vaccine_first_dose_duration_start_time = self.etiology + \
            "_vaccine_first_dose_duration_start_time"
        self.vaccine_first_dose_duration_end_time = self.etiology + \
            "_vaccine_first_dose_duration_end_time"

        self.vaccine_second_dose_duration_start_time = self.etiology + \
            "_vaccine_second_dose_duration_start_time"
        self.vaccine_second_dose_duration_end_time = self.etiology + \
            "_vaccine_second_dose_duration_end_time"

        self.vaccine_third_dose_duration_start_time = self.etiology + \
            "_vaccine_third_dose_duration_start_time"
        self.vaccine_third_dose_duration_end_time = self.etiology + \
            "_vaccine_third_dose_duration_end_time"

        self.vaccine_first_dose_working_column = self.etiology + \
            "_vaccine_first_dose_is_working"
        self.vaccine_second_dose_working_column = self.etiology + \
            "_vaccine_second_dose_is_working"
        self.vaccine_third_dose_working_column = self.etiology + \
            "_vaccine_third_dose_is_working"

        self.vaccine_unit_cost_column = self.etiology + "_vaccine_unit_cost"
        self.vaccine_cost_to_administer_column = "cost_to_administer_" + \
            self.etiology + "_vaccine"

    def setup(self, builder):

        columns = [self.vaccine_first_dose_column,
                   self.vaccine_second_dose_column,
                   self.vaccine_third_dose_column,
                   self.vaccine_unit_cost_column,
                   self.vaccine_cost_to_administer_column,
                   self.vaccine_first_dose_time_column,
                   self.vaccine_second_dose_time_column,
                   self.vaccine_third_dose_time_column,
                   self.vaccine_first_dose_count_column,
                   self.vaccine_second_dose_count_column,
                   self.vaccine_third_dose_count_column,
                   self.vaccine_first_dose_duration_start_time,
                   self.vaccine_first_dose_duration_end_time,
                   self.vaccine_second_dose_duration_start_time,
                   self.vaccine_second_dose_duration_end_time,
                   self.vaccine_third_dose_duration_start_time,
                   self.vaccine_third_dose_duration_end_time,
                   self.vaccine_first_dose_working_column,
                   self.vaccine_second_dose_working_column,
                   self.vaccine_third_dose_working_column,
                   'age']

        self.population_view = builder.population_view(columns, query='alive')


    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_count_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_count_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_count_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_time_column: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_time_column: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_time_column: [pd.NaT]*len(event.index)},
            index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_duration_start_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_duration_end_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_duration_start_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_duration_end_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_duration_start_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_duration_end_time: [pd.NaT]*len(event.index)},
            index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_working_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_working_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_working_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))

        self.population_view.update(pd.DataFrame({
            self.vaccine_unit_cost_column: np.zeros(len(event.index),
            dtype=float)}, index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_cost_to_administer_column: np.zeros(len(event.index),
            dtype=int)}, index=event.index))

    # FIXME: An emitter could potentially be faster. Could have an emitter that
    #     says when people reach a certain age, give them a vaccine dose.
    @listens_for('time_step')
    @uses_columns(['age', 'rotaviral_entiritis_vaccine_first_dose',
                   'rotaviral_entiritis_vaccine_second_dose',
                   'rotaviral_entiritis_vaccine_third_dose',
                   'rotaviral_entiritis_vaccine_first_dose_count',
                   'rotaviral_entiritis_vaccine_second_dose_count',
                   'rotaviral_entiritis_vaccine_third_dose_count',
                   'rotaviral_entiritis_vaccine_first_dose_event_time',
                   'rotaviral_entiritis_vaccine_second_dose_event_time',
                   'rotaviral_entiritis_vaccine_third_dose_event_time',
                   'rotaviral_entiritis_vaccine_unit_cost',
                   'cost_to_administer_rotaviral_entiritis_vaccine',
                   'rotaviral_entiritis_vaccine_first_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_first_dose_duration_end_time',
                   'rotaviral_entiritis_vaccine_second_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_second_dose_duration_end_time',
                   'rotaviral_entiritis_vaccine_third_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_third_dose_duration_end_time',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working'], 'alive')
    def _determine_who_gets_vaccinated(self, event):
        """
        Each time step, call the _determine_who_should_receive_dose function to
            see which patients should get dosed. We do this for all 3 vaccines
            separately.
        """
        if not self.active:
            return

        else:
            population = self.population_view.get(event.index)

            children_who_will_receive_first_dose = determine_who_should_receive_dose(
                population, event.index, self.vaccine_first_dose_column, 1)

            if not children_who_will_receive_first_dose.empty:
                children_who_will_receive_first_dose = accrue_vaccine_cost_and_count(
                    children_who_will_receive_first_dose,
                    self.vaccine_first_dose_time_column,
                    self.vaccine_first_dose_count_column,
                    self.vaccine_unit_cost_column,
                    self.vaccine_cost_to_administer_column,
                    pd.Timestamp(event.time))

                children_who_will_receive_first_dose = set_vaccine_duration(
                    children_who_will_receive_first_dose, event.time,
                    "rotaviral_entiritis", "first")

                self.population_view.update(children_who_will_receive_first_dose)

            # Second dose
            children_who_will_receive_second_dose = determine_who_should_receive_dose(
                population, event.index, self.vaccine_second_dose_column, 2)

            if not children_who_will_receive_second_dose.empty:
                children_who_will_receive_second_dose = accrue_vaccine_cost_and_count(
                    children_who_will_receive_second_dose,
                    self.vaccine_second_dose_time_column,
                    self.vaccine_second_dose_count_column,
                    self.vaccine_unit_cost_column,
                    self.vaccine_cost_to_administer_column,
                    pd.Timestamp(event.time))

                children_who_will_receive_second_dose = set_vaccine_duration(
                    children_who_will_receive_second_dose, event.time,
                    "rotaviral_entiritis", "second")

                self.population_view.update(children_who_will_receive_second_dose)

            # Third dose
            children_who_will_receive_third_dose = determine_who_should_receive_dose(
                population, event.index, self.vaccine_third_dose_column, 3)

            if not children_who_will_receive_third_dose.empty:
                children_who_will_receive_third_dose = accrue_vaccine_cost_and_count(
                    children_who_will_receive_third_dose,
                    self.vaccine_third_dose_time_column,
                    self.vaccine_third_dose_count_column,
                    self.vaccine_unit_cost_column,
                    self.vaccine_cost_to_administer_column,
                    pd.Timestamp(event.time))

                children_who_will_receive_third_dose = set_vaccine_duration(
                    children_who_will_receive_third_dose, event.time,
                    "rotaviral_entiritis", "third")

                self.population_view.update(children_who_will_receive_third_dose)

    @listens_for('time_step')
    @uses_columns(['rotaviral_entiritis_vaccine_first_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_first_dose_duration_end_time',
                   'rotaviral_entiritis_vaccine_second_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_second_dose_duration_end_time',
                   'rotaviral_entiritis_vaccine_third_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_third_dose_duration_end_time',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working'], 'alive')
    def set_working_column(self, event):
        population = self.population_view.get(event.index)

        population = _set_working_column(population, event.time, "rotaviral_entiritis")

        self.population_view.update(population)


    @modifies_value('incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    @uses_columns(['diarrhea_due_to_rotaviral_entiritis',
                   'rotaviral_entiritis_vaccine_third_dose',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working'], 'alive')
    def incidence_rates(self, index, rates, population_view):
        """
        If the intervention is running, determine who is currently receiving
        the intervention and then decrease their incidence of diarrhea due to
        rota by the effectiveness specified in the config file

        Parameters
        ----------
        index: pandas index
            index of all simulants

        rates: pd.Series
            incidence rates for diarrhea due to rotavirus

        population_view: pd.DataFrame
            dataframe of all simulants that are alive with columns
            diarrhea_due_to_rotaviral_entiritis,
            rotaviral_entiritis_vaccine_third_dose,
            rotaviral_entiritis_vaccine_is_working
        """
        population = self.population_view.get(index)

        # set up so that rates are manipulated for each working col separately
        if self.active:
            # TODO: Make this more flexible. It would be nice to be able to have
            #     this function work regardless of the number of doses

            for dose, dose_number in {"first": 1, "second": 2, "third": 3}.items():
                # TODO: Figure out how to pass etiology in as an argument here so that rotaviral entiritis isn't hardcoded into line below
                dose_working_index = population.query("rotaviral_entiritis_vaccine_{d}_dose_is_working == 1".format(d=dose)).index
                # confer full protection to people that receive 3 vaccines,
                #     partial protection to those that only receive 1 or 2
                vaccine_effectiveness = config.getfloat('rota_vaccine', '{}_dose_effectiveness'.format(dose))
                rates.loc[dose_working_index] *= (1 - vaccine_effectiveness)

            return rates

        else:
            return rates


    @modifies_value('metrics')
    @uses_columns(['rotaviral_entiritis_vaccine_first_dose_count',
                   'rotaviral_entiritis_vaccine_second_dose_count',
                   'rotaviral_entiritis_vaccine_third_dose_count',
                   'rotaviral_entiritis_vaccine_unit_cost',
                   'cost_to_administer_rotaviral_entiritis_vaccine'])
    def metrics(self, index, metrics, population_view):
        """
        Update the output metrics with information regarding the vaccine
        intervention

        Parameters
        ----------
        index: pandas Index
            Index of all simulants, alive or dead

        metrics: pd.Dictionary
            Dictionary of metrics that will be printed out at the end of the
            simulation

        population_view: pd.DataFrame
            df of all simulants, alive or dead with columns
            rotaviral_entiritis_vaccine_first_dose_count,
            rotaviral_entiritis_vaccine_second_dose_count,
            rotaviral_entiritis_vaccine_third_dose_count,
            rotaviral_entiritis_vaccine_unit_cost,
            cost_to_administer_rotaviral_entiritis_vaccine
        """

        population = population_view.get(index)

        metrics['rotaviral_entiritis_vaccine_first_dose_count'] = population[
            'rotaviral_entiritis_vaccine_first_dose_count'].sum()
        metrics['rotaviral_entiritis_vaccine_second_dose_count'] = population[
            'rotaviral_entiritis_vaccine_second_dose_count'].sum()
        metrics['rotaviral_entiritis_vaccine_third_dose_count'] = population[
            'rotaviral_entiritis_vaccine_third_dose_count'].sum()

        metrics[self.vaccine_unit_cost_column] = population[
            self.vaccine_unit_cost_column].sum()
        metrics[self.vaccine_cost_to_administer_column] = population[
            self.vaccine_cost_to_administer_column].sum()

        return metrics
