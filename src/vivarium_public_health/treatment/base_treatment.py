import numpy as np
import pandas as pd

from vivarium.framework.components import ComponentConfigError


class Treatment:

    def __init__(self, name, cause):
        self.name = name
        self.cause = cause
        self.treatment_effects = []

    def setup(self, builder):
        builder.components.add_components(self.treatment_effects)
        if self.name not in builder.configuration:
            raise ComponentConfigError(f'No configuration found for {self.name}')

        self.config = builder.configuration[self.name]
        self.dose_response = dict(
            onset_delay=pd.to_timedelta(self.config.dose_response.onset_delay, unit='D'),
            duration=pd.to_timedelta(self.config.dose_response.duration, unit='D'),
            waning_rate=float(self.config.dose_response.waning_rate),
        )
        self.protection = self._get_protection(builder)

        builder.value.register_value_modifier(f'{self.cause}.incidence_rate',
                                              modifier=self.incidence_rates)

        columns = [f'{self.name}_current_dose',
                   f'{self.name}_current_dose_event_time',
                   f'{self.name}_previous_dose',
                   f'{self.name}_previous_dose_event_time']
        self.population_view = builder.population.get_view(['alive']+columns)

        self.clock = builder.time.clock()

    def _get_protection(self, builder):
        return self.get_protection(builder)

    @staticmethod
    def get_protection(builder):
        raise NotImplementedError('You must supply an implementation of get_protection')

    def _get_dosing_status(self, population):
        received_current_dose = population[f'{self.name}_current_dose'].notnull()
        current_dose_full_immunity_start = (population[f'{self.name}_current_dose_event_time']
                                            + self.dose_response['onset_delay'])
        current_dose_giving_immunity = received_current_dose & (current_dose_full_immunity_start <= self.clock())

        received_previous_dose = population[f'{self.name}_previous_dose'].notna()
        previous_dose_full_immunity_start = (population[f'{self.name}_previous_dose_event_time']
                                             + self.dose_response['onset_delay'])
        previous_dose_giving_immunity = (received_previous_dose
                                         & (previous_dose_full_immunity_start <= self.clock())
                                         & ~current_dose_giving_immunity)

        dosing_status = pd.DataFrame({'dose': None, 'date': pd.NaT}, index=population.index)
        #  not sure why, but pandas doesn't save the sliced data for two columns at the same time
        dosing_status.loc[current_dose_giving_immunity, 'dose'] = population.loc[
            current_dose_giving_immunity, f'{self.name}_current_dose']
        dosing_status.loc[current_dose_giving_immunity, 'date'] = population.loc[
            current_dose_giving_immunity, f'{self.name}_current_dose_event_time']
        dosing_status.loc[previous_dose_giving_immunity, 'dose'] = population.loc[
            previous_dose_giving_immunity, f'{self.name}_previous_dose']
        dosing_status.loc[previous_dose_giving_immunity, 'date'] = population.loc[
            previous_dose_giving_immunity, f'{self.name}_previous_dose_event_time']

        return dosing_status

    def determine_protection(self, population):
        """Determines how much protection simulants receive from the vaccine.

         Parameters
         ----------
         population: pandas.DataFrame
             A copy of the relevant portions of the simulation state table.

         Returns
         -------
         pandas.Series
             A list of the protection conferred by the vaccine indexed by the simulant ids.

         every time step, we have 3 types of immunities:
         1. no immunity
            a. Never get any dose (filter from got_this_dose)
            b. Got the most recent dose as 1st dose but still in delay
         2. waning immunity
            a. full immunity ended from the most recent dose
            b. full immunity ended from the previous dose, got a new dose but still in delay
         3. full immunity
            a. have full immunity from the most recent dose
            b. Got the previous dose and most recent dose: full immunity from previous dose/ most recent dose in delay
         """
        dosing_status = self._get_dosing_status(population)

        no_immunity = dosing_status['dose'].isnull()
        full_immunity = self.clock() < (dosing_status['date'] + self.dose_response['onset_delay']
                                        + self.dose_response['duration'])
        waning_immunity = ~no_immunity & ~full_immunity
        time_in_waning = self.clock() - (dosing_status.loc[waning_immunity, 'date']
                                         + self.dose_response['onset_delay'] + self.dose_response['duration'])

        protection = pd.Series(0, index=population.index)

        protection[full_immunity | waning_immunity] = dosing_status.dose[
            full_immunity | waning_immunity].map(self.protection)

        protection[waning_immunity] *= np.exp(-self.dose_response['waning_rate']*time_in_waning.dt.days)

        return protection

    def incidence_rates(self, index, rates):
        """Modifies the incidence of shigellosis.

        Parameters
        ----------
        index: pandas.Index
            The set of simulants who are susceptible to shigellosis.
        rates: pandas.Series
            The baseline incidence rate of shigellosis.

        Returns
        -------
        pandas.Series
            The shigellosis incidence rates adjusted for the presence of the vaccine.
        """
        population = self.population_view.get(index)
        population = population[population.alive == 'alive']
        protection = self.determine_protection(population)
        rates.loc[population.index] *= (1-protection.values)
        return rates
