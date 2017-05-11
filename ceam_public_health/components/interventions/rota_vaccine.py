import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs import get_rota_vaccine_coverage


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
    population[vaccine_unit_cost_column] += config.rota_vaccine.RV5_dose_cost

    population[vaccine_cost_to_administer_column] += config.rota_vaccine.cost_to_administer_each_dose

    return population


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
    time_after_dose_at_which_immunity_is_conferred = config.rota_vaccine.time_after_dose_at_which_immunity_is_conferred

    population["{e}_vaccine_{d}_dose_duration_start_time".format(e=etiology, d=dose)] = \
        population["{e}_vaccine_{d}_dose_event_time".format(e=etiology, d=dose)] + \
        pd.to_timedelta(time_after_dose_at_which_immunity_is_conferred, unit='D')
    
    # determine when the effect of the vaccine should end
    vaccine_duration = config.rota_vaccine.vaccine_duration
    waning_immunity_time = config.rota_vaccine.waning_immunity_time

    if waning_immunity_time == 0:
        population["{e}_vaccine_{d}_dose_duration_end_time".format(e=etiology, d=dose)] = \
            population["{e}_vaccine_{d}_dose_duration_start_time".format(e=etiology, d=dose)] + \
            pd.to_timedelta(vaccine_duration, unit='D')

    if waning_immunity_time != 0:
        population["{e}_vaccine_{d}_dose_duration_end_time".format(e=etiology, d=dose)] = \
            population["{e}_vaccine_{d}_dose_duration_start_time".format(e=etiology, d=dose)] + \
            pd.to_timedelta(vaccine_duration, unit='D') + pd.to_timedelta(waning_immunity_time, unit='D')
        
    return population

# FIXME: Do not need the dose working index anymore. Should start to write code to move away from this. 
#     Can just set up the code so that effectiveness is either 0 or gt than
#     0, depending on whether the simulant was vaccinated
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
        population["{e}_vaccine_{d}_dose_is_working".format(e=etiology, d=dose)] = 0
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

# TODO: Results changed a bit when this function was added. Confirm that the function is working correctly
def wane_immunity(days, duration, vaccine_waning_time, vaccine_effectiveness):
    """
    Create waning immunity function. This function returns a univariate spline that can be called to get an effectiveness estimate
    when supplied a certain number of days since vaccination
    
    Parameters
    ----------
    days: int
        number of days since vaccine duration start date

    duration: int
        number of days that the vaccine will last at full effectiveness

    vaccine_waning_time: int
        number of days vaccine will wane

    vaccine_effectiveness: float
        reduction in incidence as a result of receiving the vaccine
    """
    # FIXME: Probably a better way to 735 as the end point of vaccine effectiveness 
    x = [0, duration] + [duration + vaccine_waning_time  + .00001]
    y = [vaccine_effectiveness, vaccine_effectiveness] + [0]
    # set the order to 1 (linear), s to 0 (sum of least squares=0), ext to 1 (all extrapolated values are 0)
    spl = UnivariateSpline(x, y, k=1, s=0, ext=1)
    return spl(days)


def determine_vaccine_effectiveness(pop, dose_working_index, waning_immunity_function, current_time, dose, duration, vaccine_waning_time, vaccine_effectiveness):
    """
    Determine the effectiveness of a vaccine based on how many days its been since the simulant received the vaccine
    
    Parameters
    ----------
    pop: pd.DataFrame
        population_view of simulants
        
    dose_working_index: pandas index
        index of simulants for whom the current dose is working
    
    waning_immunity_function: UnivariateSpline object
        scipy.interpolat.UnivariateSpline object containing estimates of vaccine effectiveness (Y) given number of days since vaccination (X)
        
    current_time: datetime.datetime
        current time in the simulation
        
    dose: str
        can be one of "first", "second", or "third"

    vaccine_waning_time: int
        number of days vaccine will wane

    vaccine_effectiveness: float
        effectiveness of the current dose of the vaccine
    """
    pop['days_since_vaccine_started_working'] = current_time - \
    pop['rotaviral_entiritis_vaccine_{}_dose_duration_start_time'.format(dose)]
    
    pop = pop[pop.days_since_vaccine_started_working.notnull()]
    
    pop['days_since_vaccine_started_working'] = (pop['days_since_vaccine_started_working'] / np.timedelta64(1, 'D')).astype(int)
    
    pop['effectiveness'] = pop['days_since_vaccine_started_working'].apply(lambda days: waning_immunity_function(days, duration, vaccine_waning_time, vaccine_effectiveness))
    
    return pop.loc[dose_working_index]['effectiveness']


class RotaVaccine():
    """
    Class that determines who gets vaccinated, how the vaccine affects
    incidence, and counts vaccinations
    """
    configuration_defaults = {
            'rota_vaccine': {
                'RV5_dose_cost': 3.5,
                'cost_to_administer_each_dose': 0,
                'first_dose_effectiveness': 0,
                'second_dose_effectiveness': 0,
                'third_dose_effectiveness': .39,
                'vaccine_duration': 730,
                'waning_immunity_time': 0,
                'age_at_first_dose': 61,
                'age_at_second_dose': 122,
                'age_at_third_dose': 183,
                'second_dose_retention': 1,
                'third_dose_retention': 1,
                'vaccination_proportion_increase': .5,
                'time_after_dose_at_which_immunity_is_conferred': 14,
            }
    }

    def __init__(self, active):
        self.active = active
        self.etiology = 'rotaviral_entiritis'
        self.etiology_column = self.etiology

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
                   'fractional_age']

        self.clock = builder.clock()
        self.population_view = builder.population_view(columns, query='alive')

        self.randomness_dict = {}
        self.randomness_dict['dose_1'] = builder.randomness('first_dose_randomness')
        self.randomness_dict['dose_2'] = builder.randomness('second_dose_randomness')
        self.randomness_dict['dose_3'] = builder.randomness('third_dose_randomness')

        self.vaccine_coverage = builder.value('{}_vaccine_coverage'.format(self.etiology))
        self.vaccine_coverage.source = builder.lookup(get_rota_vaccine_coverage())

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

    def _determine_who_should_receive_dose(self, population, vaccine_col, true_weight,
                                           dose_age, dose_number):
        """
        Uses choice to determine if each simulant should receive a dose. Returns a
            population of simulants that should receive a dose (most of the time
            this function will return an empty population)

        Parameters
        ----------
        population: pd.DataFrame
            population view of all of the simulants who are currently alive

        vaccine_col: str
            str representing the name of a column, one of
            rotaviral_entiritis_vaccine_first_dose, second, or third dose.
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

        if dose_number == 1:
            children_at_dose_age = population.query(
                "age_in_days == @dose_age").copy()

        elif dose_number == 2:
            children_at_dose_age = population.query(
                "age_in_days == @dose_age and" +
                " rotaviral_entiritis_vaccine_first_dose == 1").copy()

        elif dose_number == 3:
            children_at_dose_age = population.query(
                "age_in_days == @dose_age and" +
                " rotaviral_entiritis_vaccine_second_dose == 1").copy()

        else:
            raise(ValueError, "dose_number cannot be any value other than" +
                              " 1, 2, or 3")

        if not children_at_dose_age.empty:
            children_at_dose_age[vaccine_col] = self.randomness_dict['dose_{}'.format(dose_number)].choice(
                children_at_dose_age.index, [1]*len(true_weight) + [0]*len(false_weight), true_weight.tolist() + false_weight.tolist())

        children_who_will_receive_dose = children_at_dose_age.query(
            "{} == 1".format(vaccine_col))

        return children_who_will_receive_dose


    def determine_who_should_receive_dose(self, population, index, vaccine_col,
                                          dose_number, current_time):
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

        current_time:
            current_time in the simulation
        """

        population['age_in_days'] = population['fractional_age'] * 365

        population['age_in_days'] = population['age_in_days'].round()

        # FIXME: Want for true vaccine coverage to occurr in baseline (active = False) scenario
        vaccine_coverage = self.vaccine_coverage(population.index)

        # FIXME: GBD coverage metric is a measure of people that receive all 3 vaccines, not just 1.
        #     Need to figure out a way to capture this in the model
        if dose_number == 1:
            true_weight =  vaccine_coverage + config.rota_vaccine.vaccination_proportion_increase
            dose_age = config.rota_vaccine.age_at_first_dose

        if dose_number == 2:
            true_weight = pd.Series(config.rota_vaccine.second_dose_retention, index=population.index)
            dose_age = config.rota_vaccine.age_at_second_dose

        if dose_number == 3:
            true_weight = pd.Series(config.rota_vaccine.third_dose_retention, index=population.index)
            dose_age = config.rota_vaccine.age_at_third_dose

        children_who_will_receive_dose = self._determine_who_should_receive_dose(
            population=population, vaccine_col=vaccine_col,
            true_weight=true_weight, dose_age=dose_age, dose_number=dose_number)

        return children_who_will_receive_dose


    # FIXME: An emitter could potentially be faster. Could have an emitter that
    #     says when people reach a certain age, give them a vaccine dose.
    @listens_for('time_step')
    @uses_columns(['fractional_age', 'rotaviral_entiritis_vaccine_first_dose',
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
            population = event.population

            children_who_will_receive_first_dose = self.determine_who_should_receive_dose(
                population, event.index, self.vaccine_first_dose_column, 1, event.time)

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

                event.population_view.update(children_who_will_receive_first_dose)

            # Second dose
            children_who_will_receive_second_dose = self.determine_who_should_receive_dose(
                population, event.index, self.vaccine_second_dose_column, 2, event.time)

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

                event.population_view.update(children_who_will_receive_second_dose)

            # Third dose
            children_who_will_receive_third_dose = self.determine_who_should_receive_dose(
                population, event.index, self.vaccine_third_dose_column, 3, event.time)

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

                event.population_view.update(children_who_will_receive_third_dose)

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
        population = event.population

        population = _set_working_column(population, event.time, "rotaviral_entiritis")

        event.population_view.update(population)


    @modifies_value('incidence_rate.rotaviral_entiritis')
    @uses_columns(['rotaviral_entiritis',
                   'rotaviral_entiritis_vaccine_third_dose',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working',
                   'rotaviral_entiritis_vaccine_first_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_second_dose_duration_start_time',
                   'rotaviral_entiritis_vaccine_third_dose_duration_start_time'], 'alive')
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
            rotaviral_entiritis,
            rotaviral_entiritis_vaccine_third_dose,
            rotaviral_entiritis_vaccine_is_working
        """
        population = population_view.get(index)

        # start with refactoring, then move to looking at scalar vs. constant spline
        # set up so that rates are manipulated for each working col separately
        if self.active:
            # TODO: Make this more flexible. It would be nice to be able to have
            #     this function work regardless of the number of doses
            for dose, dose_number in {"first": 1, "second": 2, "third": 3}.items():
                # TODO: Figure out how to pass etiology in as an argument here so that rotaviral entiritis isn't hardcoded into line below
                dose_working_index = population.query("rotaviral_entiritis_vaccine_{d}_dose_is_working == 1".format(d=dose)).index
                # confer full protection to people that receive 3 vaccines,
                #     partial protection to those that only receive 1 or 2
                duration = config.rota_vaccine.vaccine_duration

                # FIXME: I feel like there should be a better way to get effectiveness using the new config, but I don't know how. Using the old config, I could say config.getfloat('rota_vaccine', '{}_dose_effectiveness'.format(dose))
                if dose == "first":
                    effectiveness =  config.rota_vaccine.first_dose_effectiveness
                if dose == "second":
                    effectiveness =  config.rota_vaccine.second_dose_effectiveness
                if dose == "third":
                    effectiveness =  config.rota_vaccine.third_dose_effectiveness

                waning_immunity_time = config.rota_vaccine.waning_immunity_time

                if not len(dose_working_index) == 0:
                    vaccine_effectiveness = determine_vaccine_effectiveness(population, dose_working_index, wane_immunity, self.clock(), dose, duration, waning_immunity_time, effectiveness)
                else:
                    vaccine_effectiveness = 0
                # TODO: Confirm whether this affects rates or probabilities
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
