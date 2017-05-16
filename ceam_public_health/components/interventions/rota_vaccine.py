import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam.framework.randomness import filter_for_probability

from ceam_inputs import get_rota_vaccine_coverage


def set_vaccine_duration(population, etiology, dose):
    """
    Function that sets vaccine dose immunity start/end time
    
    Parameters
    ----------
    population : pd.DataFrame()
        population_view of simulants that have just been vaccinated
        
    etiology: str
        specific etiology that is being afftected by the vaccine.
        in the case of rota, our etiology is rotaviral entiritis
    
    dose: str
        can be "first", "second", or "third"
    """
    assert dose in ["first", "second", "third"], "dose can be one of first, second, or third"
    
    # determine when effect of the vaccine should start
    time_after_dose_at_which_immunity_is_conferred = config.rota_vaccine.time_after_dose_at_which_immunity_is_conferred

    population["{e}_vaccine_{d}_dose_immunity_start_time".format(e=etiology, d=dose)] = \
        population["{e}_vaccine_{d}_dose_event_time".format(e=etiology, d=dose)] + \
        pd.to_timedelta(time_after_dose_at_which_immunity_is_conferred, unit='D')
 
    # determine when the effect of the vaccine should end
    vaccine_full_immunity_duration = config.rota_vaccine.vaccine_full_immunity_duration
    waning_immunity_time = config.rota_vaccine.waning_immunity_time

    population["{e}_vaccine_{d}_dose_immunity_end_time".format(e=etiology, d=dose)] = \
        population["{e}_vaccine_{d}_dose_immunity_start_time".format(e=etiology, d=dose)] + \
        pd.to_timedelta(vaccine_full_immunity_duration, unit='D') + pd.to_timedelta(waning_immunity_time, unit='D')
        
    return population


def wane_immunity(days, full_immunity_duration, vaccine_waning_time, vaccine_effectiveness):
    """
    Create waning immunity function. This function returns a univariate spline that can be called to get an effectiveness estimate
    when supplied a certain number of days since the dose's immunity start time
    
    Parameters
    ----------
    days: int
        number of days since vaccine duration start date

    full_immunity_duration: int
        number of days that the vaccine will last at full effectiveness

    vaccine_waning_time: int
        number of days vaccine will wane

    vaccine_effectiveness: float
        reduction in incidence as a result of receiving the vaccine
    """
    x = [0, full_immunity_duration, full_immunity_duration + vaccine_waning_time  + .00001]
    y = [vaccine_effectiveness, vaccine_effectiveness, 0]

    # set the order to 1 (linear), s to 0 (sum of least squares=0), ext to 1 (all extrapolated values are 0)
    spl = UnivariateSpline(x, y, k=1, s=0, ext=1)
    return spl(days)


def determine_vaccine_protection(pop, dose_working_index, waning_immunity_function, current_time, dose, vaccine_effectiveness):
    """
    Determine the protection of a vaccine based on how many days its been since the simulant received the vaccine
    
    Parameters
    ----------
    pop: pd.DataFrame
        population_view of simulants
        
    dose_working_index: pandas index
        index of simulants for whom the current dose is working
    
    waning_immunity_function: UnivariateSpline object
        function that returns a scipy.interpolate.UnivariateSpline object containing estimates of vaccine effectiveness (Y) given number of days since vaccine immunity start time (X)
        
    current_time: datetime.datetime
        current time in the simulation
        
    dose: str
        can be one of "first", "second", or "third"

    vaccine_effectiveness: float
        effectiveness of the current dose of the vaccine
    """
    pop['days_since_vaccine_started_conferring_immunity'] = current_time - \
    pop['rotaviral_entiritis_vaccine_{}_dose_immunity_start_time'.format(dose)]
    
    pop = pop[pop.days_since_vaccine_started_conferring_immunity.notnull()]
    
    pop['days_since_vaccine_started_conferring_immunity'] = (pop['days_since_vaccine_started_conferring_immunity'] / np.timedelta64(1, 'D')).astype(int)

    full_immunity_duration = config.rota_vaccine.vaccine_full_immunity_duration

    waning_immunity_time = config.rota_vaccine.waning_immunity_time

    pop['effectiveness'] = pop['days_since_vaccine_started_conferring_immunity'].apply(lambda days: waning_immunity_function(days, full_immunity_duration, waning_immunity_time, vaccine_effectiveness))
    
    return pop.loc[dose_working_index]['effectiveness']


class RotaVaccine():
    """
    RotaVaccine accomplishes several tasks 
        1) We administer the vaccine. We use determine which simulants should receive a dose of the vaccine, set the time at which the received the vaccine, and set the vaccine full immunity duration + waning immunity time 
        2) We set a 'working col' for each dose of the vaccine, to determine which whether or not the simulant should be experiencing any protection from the vaccine
        3) We then use the estimate of vaccine protection to reduce the incidence of rotaviral entiritis infection. The protection conferred by the vaccine depends on how long ago the vaccine was administered and how many doses a simulant has received
        4) Finally, we output vaccine cost/count metrics
    """
    
    configuration_defaults = {
            'rota_vaccine': {
                'RV5_dose_cost': 3.5,
                'cost_to_administer_each_dose': 0,
                'first_dose_effectiveness': 0,
                'second_dose_effectiveness': 0,
                'third_dose_effectiveness': .39,
                'vaccine_full_immunity_duration': 730,
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

        self.vaccine_column = self.etiology + "_vaccine"

        self.vaccine_first_dose_time_column = self.etiology + \
            "_vaccine_first_dose_event_time"
        self.vaccine_second_dose_time_column = self.etiology + \
            "_vaccine_second_dose_event_time"
        self.vaccine_third_dose_time_column = self.etiology + \
            "_vaccine_third_dose_event_time"

        self.vaccine_first_dose_immunity_start_time = self.etiology + \
            "_vaccine_first_dose_immunity_start_time"
        self.vaccine_first_dose_immunity_end_time = self.etiology + \
            "_vaccine_first_dose_immunity_end_time"

        self.vaccine_second_dose_immunity_start_time = self.etiology + \
            "_vaccine_second_dose_immunity_start_time"
        self.vaccine_second_dose_immunity_end_time = self.etiology + \
            "_vaccine_second_dose_immunity_end_time"

        self.vaccine_third_dose_immunity_start_time = self.etiology + \
            "_vaccine_third_dose_immunity_start_time"
        self.vaccine_third_dose_immunity_end_time = self.etiology + \
            "_vaccine_third_dose_immunity_end_time"

        self.vaccine_first_dose_working_column = self.etiology + \
            "_vaccine_first_dose_is_working"
        self.vaccine_second_dose_working_column = self.etiology + \
            "_vaccine_second_dose_is_working"
        self.vaccine_third_dose_working_column = self.etiology + \
            "_vaccine_third_dose_is_working"


    def setup(self, builder):

        columns = [self.vaccine_column,
                   self.vaccine_first_dose_time_column,
                   self.vaccine_second_dose_time_column,
                   self.vaccine_third_dose_time_column,
                   self.vaccine_first_dose_immunity_start_time,
                   self.vaccine_first_dose_immunity_end_time,
                   self.vaccine_second_dose_immunity_start_time,
                   self.vaccine_second_dose_immunity_end_time,
                   self.vaccine_third_dose_immunity_start_time,
                   self.vaccine_third_dose_immunity_end_time,
                   self.vaccine_first_dose_working_column,
                   self.vaccine_second_dose_working_column,
                   self.vaccine_third_dose_working_column]

        self.clock = builder.clock()
        self.population_view = builder.population_view(columns, query='alive')

        self.randomness = {}
        self.randomness['dose_1'] = builder.randomness('first_dose_randomness')
        self.randomness['dose_2'] = builder.randomness('second_dose_randomness')
        self.randomness['dose_3'] = builder.randomness('third_dose_randomness')

        self.vaccine_coverage = builder.value('{}_vaccine_coverage'.format(self.etiology))
        self.vaccine_coverage.source = builder.lookup(get_rota_vaccine_coverage())

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({
            self.vaccine_column: np.zeros(len(event.index),
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
            self.vaccine_first_dose_immunity_start_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_first_dose_immunity_end_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_immunity_start_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_second_dose_immunity_end_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_immunity_start_time: [pd.NaT]*len(event.index)},
            index=event.index))
        self.population_view.update(pd.DataFrame({
            self.vaccine_third_dose_immunity_end_time: [pd.NaT]*len(event.index)},
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


    def determine_who_should_receive_dose(self, population, vaccine_col,
                                           dose_number):
        """
        Uses choice to determine if each simulant should receive a dose. Returns a
            population of simulants that should receive a dose of a vaccine

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
            vaccine that is currently being administered

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
        population['age_in_days'] = population['fractional_age'] * 365

        population['age_in_days'] = population['age_in_days'].round()

        vaccine_coverage = self.vaccine_coverage(population.index)

        previous_dose = dose_number - 1

        # FIXME: GBD coverage metric is a measure of people that receive all 3 vaccines, not just 1.
        #     Need to figure out a way to capture this in the model
        if dose_number == 1:
            dose_age = config.rota_vaccine.age_at_first_dose
            children_at_dose_age = population.query(
                "age_in_days == @dose_age").copy()
            if self.active:
                true_weight =  vaccine_coverage + config.rota_vaccine.vaccination_proportion_increase
            else:
                true_weight = vaccine_coverage

        elif dose_number == 2:
            dose_age = config.rota_vaccine.age_at_second_dose
            children_at_dose_age = population.query(
                "age_in_days == @dose_age and" +
                " rotaviral_entiritis_vaccine == {pd}".format(pd=previous_dose)).copy()
            true_weight = pd.Series(config.rota_vaccine.second_dose_retention, index=children_at_dose_age.index)

        elif dose_number == 3:
            dose_age = config.rota_vaccine.age_at_third_dose
            children_at_dose_age = population.query(
                "age_in_days == @dose_age and" +
                " rotaviral_entiritis_vaccine == {pd}".format(pd=previous_dose)).copy()
            true_weight = pd.Series(config.rota_vaccine.third_dose_retention, index=children_at_dose_age.index)

        else:
            raise(ValueError, "dose_number cannot be any value other than" +
                              " 1, 2, or 3")

        false_weight = 1 - true_weight

        children_who_will_receive_dose_index = pd.Index

        if not children_at_dose_age.empty:
            children_who_will_receive_dose_index = self.randomness['dose_{}'.format(dose_number)].filter_for_probability(
                children_at_dose_age.index, true_weight)

        children_who_will_receive_dose = children_at_dose_age.loc[children_who_will_receive_dose_index]

        children_who_will_receive_dose[vaccine_col] = dose_number

        return children_who_will_receive_dose


    @listens_for('time_step')
    @uses_columns(['fractional_age', 'rotaviral_entiritis_vaccine',
                   'rotaviral_entiritis_vaccine_first_dose_event_time',
                   'rotaviral_entiritis_vaccine_second_dose_event_time',
                   'rotaviral_entiritis_vaccine_third_dose_event_time',
                   'rotaviral_entiritis_vaccine_first_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_first_dose_immunity_end_time',
                   'rotaviral_entiritis_vaccine_second_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_second_dose_immunity_end_time',
                   'rotaviral_entiritis_vaccine_third_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_third_dose_immunity_end_time',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working'], 'alive')
    def administer_vaccine(self, event):
        """
        Each time step, call the determine_who_should_receive_dose function to
            see which patients should get dosed. We do this for all 3 vaccines
            separately. Administer_vaccine determines who will get dosed, sets
            the time at which each dose is administered, and calls
            set_vaccine_duration to set up the time at which the vaccine will
            start/end to confer immunity.
        """
        population = event.population

        children_who_will_receive_first_dose = self.determine_who_should_receive_dose(
            population, self.vaccine_column, 1)

        if not children_who_will_receive_first_dose.empty:
            children_who_will_receive_first_dose[self.vaccine_first_dose_time_column] = event.time

            children_who_will_receive_first_dose = set_vaccine_duration(
                children_who_will_receive_first_dose, 
                "rotaviral_entiritis", "first")

            event.population_view.update(children_who_will_receive_first_dose)

        # Second dose
        children_who_will_receive_second_dose = self.determine_who_should_receive_dose(
            population, self.vaccine_column, 2)

        if not children_who_will_receive_second_dose.empty:
            children_who_will_receive_second_dose[self.vaccine_second_dose_time_column] = event.time

            children_who_will_receive_second_dose = set_vaccine_duration(
                children_who_will_receive_second_dose,
                "rotaviral_entiritis", "second")

            event.population_view.update(children_who_will_receive_second_dose)

        # Third dose
        children_who_will_receive_third_dose = self.determine_who_should_receive_dose(
            population, self.vaccine_column, 3)

        if not children_who_will_receive_third_dose.empty:
            children_who_will_receive_third_dose[self.vaccine_third_dose_time_column] = event.time

            children_who_will_receive_third_dose = set_vaccine_duration(
                children_who_will_receive_third_dose, 
                "rotaviral_entiritis", "third")

            event.population_view.update(children_who_will_receive_third_dose)


    @listens_for('time_step')
    @uses_columns(['rotaviral_entiritis_vaccine_first_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_first_dose_immunity_end_time',
                   'rotaviral_entiritis_vaccine_second_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_second_dose_immunity_end_time',
                   'rotaviral_entiritis_vaccine_third_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_third_dose_immunity_end_time',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working'], 'alive')
    def set_working_column(self, event):
        """
        Function that sets the "working column", a binary column that indicates whether the vaccine is working (1) or not working (0).
        A vaccine will only be working after it has been administered and if the current time is in between the vaccine immunity start
        and end time and the next vaccine isn't working. If the next vaccine is working, then we want to use the effect of the next 
        vaccine, so we don't want the previous vaccine to be working. For instance, if a simulant has received 2 doses of a vaccine, 
        we want for the benefit of 2 doses, not one dose, to be conferred to the simulant
        """
        population = event.population

        for dose in ["first", "second", "third"]:

            # set the working col to 0 for now, we'll set the col for some simulants to 1 below
            population["rotaviral_entiritis_vaccine_{d}_dose_is_working".format(d=dose)] = 0

            population.loc[(event.time >= population[
                "rotaviral_entiritis_vaccine_{d}_dose_immunity_start_time".format(d=dose)]) 
                & (event.time <= population[
                "rotaviral_entiritis_vaccine_{d}_dose_immunity_end_time".format(d=dose)]),
                "rotaviral_entiritis_vaccine_{d}_dose_is_working".format(d=dose)] = 1

        # now make sure that the working col is 1 only for the most recently administered vaccine
        # if third dose has been administered, set the first and second working cols to 0
        population.loc[population["rotaviral_entiritis_vaccine_third_dose_is_working"] == 1, "rotaviral_entiritis_vaccine_first_dose_is_working"] = 0
        population.loc[population["rotaviral_entiritis_vaccine_third_dose_is_working"] == 1, "rotaviral_entiritis_vaccine_second_dose_is_working"] = 0

        # if the second dose has been administered, set the first working col to 0
        population.loc[population["rotaviral_entiritis_vaccine_second_dose_is_working"] == 1, "rotaviral_entiritis_vaccine_first_dose_is_working"] = 0

        event.population_view.update(population)


    @modifies_value('incidence_rate.rotaviral_entiritis')
    @uses_columns(['rotaviral_entiritis',
                   'rotaviral_entiritis_vaccine_first_dose_is_working',
                   'rotaviral_entiritis_vaccine_second_dose_is_working',
                   'rotaviral_entiritis_vaccine_third_dose_is_working',
                   'rotaviral_entiritis_vaccine_first_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_second_dose_immunity_start_time',
                   'rotaviral_entiritis_vaccine_third_dose_immunity_start_time'], 'alive')
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

        for dose, dose_number in {"first": 1, "second": 2, "third": 3}.items():
            dose_working_index = population.query("rotaviral_entiritis_vaccine_{d}_dose_is_working == 1".format(d=dose)).index
            
            # FIXME: I feel like there should be a better way to get effectiveness using the new config, but I don't know how. Using the old config, I could say config.getfloat('rota_vaccine', '{}_dose_effectiveness'.format(dose))
            if dose == "first":
                effectiveness =  config.rota_vaccine.first_dose_effectiveness
            if dose == "second":
                effectiveness =  config.rota_vaccine.second_dose_effectiveness
            if dose == "third":
                effectiveness =  config.rota_vaccine.third_dose_effectiveness

                vaccine_protection = determine_vaccine_protection(population, dose_working_index, wane_immunity, self.clock(), dose, effectiveness)
            else:
                vaccine_protection = 0

            # TODO: Confirm whether this affects rates or probabilities
            rates.loc[dose_working_index] *= (1 - vaccine_protection)

        return rates


    @modifies_value('metrics')
    @uses_columns(['rotaviral_entiritis_vaccine'])
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
            rotaviral_entiritis_vaccine
        """

        population = population_view.get(index)

        count_vacs = population.groupby('rotaviral_entiritis_vaccine').size()

        metrics['rotaviral_entiritis_vaccine_first_dose_count'] = count_vacs[1] + count_vacs[2] + count_vacs[3]
        metrics['rotaviral_entiritis_vaccine_second_dose_count'] = count_vacs[2] + count_vacs[3]
        metrics['rotaviral_entiritis_vaccine_third_dose_count'] = count_vacs[3]

        total_number_of_administered_vaccines = metrics['rotaviral_entiritis_vaccine_first_dose_count'] + metrics['rotaviral_entiritis_vaccine_second_dose_count'] + metrics['rotaviral_entiritis_vaccine_third_dose_count']

        metrics['vaccine_unit_cost_column'] = (total_number_of_administered_vaccines) * config.rota_vaccine.RV5_dose_cost
        metrics['vaccine_cost_to_administer_column'] = (total_number_of_administered_vaccines) * config.rota_vaccine.cost_to_administer_each_dose

        return metrics
