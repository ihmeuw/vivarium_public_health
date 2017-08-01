import pandas as pd
import numpy as np

from vivarium.framework.event import listens_for
from vivarium.framework.values import modifies_value
from vivarium import config

from ceam_public_health.util import make_cols_demographically_specific, make_age_bin_age_group_max_dict


class CalculateIncidence:
    def __init__(self, disease_col, disease, disease_states):
        """
        disease_col: str
            name of the column name that contains the disease state of interest
        disease: str
            name of the disease of interest
        disease_states: list
            list of states that denote a simulant as having the disease (e.g.
            ['severe_diarrhea', 'moderate_diarrhea', 'mild_diarrhea']).
            If a simulant does not have the disease of interest, we say that they are in the susceptible state
        """
        self.disease_col = disease_col
        self.disease = disease
        self.disease_time_col = disease + "_event_time"
        self.disease_states = disease_states
        self.collecting = False
        self.incidence_rate_df = pd.DataFrame({})

        self.susceptible_person_time_cols = make_cols_demographically_specific("susceptible_person_time",
                                                                               age_group_id_min=2,
                                                                               age_group_id_max=21)
        self.event_count_cols = make_cols_demographically_specific("{}_event_count".format(self.disease),
                                                                   age_group_id_min=2,
                                                                   age_group_id_max=21)
        self.age_bin_age_group_max_dict = make_age_bin_age_group_max_dict(age_group_id_min=2,
                                                                          age_group_id_max=21)

    def setup(self, builder):
        self.clock = builder.clock()
        columns = [self.disease_col, self.disease_time_col, "exit_time", "age", "sex", "alive"]
        self.population_view = builder.population_view(columns)

    @listens_for('initialize_simulants')
    def update_incidence_rate_df(self, event):
        if self.collecting:
            new_sims = pd.DataFrame([])
            for col in self.susceptible_person_time_cols:
                new_sims[col] = pd.Series(np.zeros(len(event.index)), index=event.index)
            for col in self.event_count_cols:
                new_sims[col] = pd.Series(np.zeros(len(event.index)), index=event.index)
            self.incidence_rate_df = self.incidence_rate_df.append(new_sims)

    @listens_for('begin_epidemiological_measure_collection')
    def set_flag(self, event):
        """
        Set the collecting flag to True during GBD years
        """
        self.collecting = True
        for col in self.susceptible_person_time_cols:
            self.incidence_rate_df[col] = pd.Series(np.zeros(len(event.index)), index=event.index)
        for col in self.event_count_cols:
            self.incidence_rate_df[col] = pd.Series(np.zeros(len(event.index)), index=event.index)

    @listens_for('time_step', priority=9)
    def get_counts_and_susceptible_person_time(self, event):
        """
        Gather all of the data we need for the incidence rate calculations (event counts and susceptible person time)
        """
        if self.collecting:
            succeptible_time = config.simulation_parameters.time_step / 365

            population = self.population_view.get(event.index)
            pop = population[(population['alive'] == 'alive') | (population['exit_time'] == event.time)]

            just_exited = pop['exit_time'] == event.time
            sick = pop[self.disease_col].isin(self.disease_states)
            got_sick_this_time_step = pop[self.disease_time_col] == event.time

            for sex in ["Male", "Female"]:
                last_age_group_max = 0
                for age_bin, upr_bound in self.age_bin_age_group_max_dict:
                    appropriate_age_and_sex = ((pop['age'] < upr_bound)
                                               & (pop['age'] >= last_age_group_max)
                                               & (pop['sex'] == sex))

                    event_count_column = '{}_event_count_{}_among_{}s'.format(self.disease, age_bin, sex)
                    succeptible_time_column = 'susceptible_person_time_{}_among_{}s'.format(age_bin, sex)

                    cases_index = pop[appropriate_age_and_sex & sick & got_sick_this_time_step].index
                    susceptible_index = pop[~sick & appropriate_age_and_sex].index
                    just_exited_index = pop[~sick & appropriate_age_and_sex & just_exited].index

                    self.incidence_rate_df[event_count_column].loc[cases_index] += 1
                    self.incidence_rate_df[succeptible_time_column].loc[susceptible_index] += succeptible_time
                    self.incidence_rate_df[succeptible_time_column].loc[just_exited_index] += succeptible_time / 2

                    last_age_group_max = upr_bound

    @modifies_value('epidemiological_span_measures')
    def calculate_incidence_measure(self, index, age_groups, sexes, all_locations, duration, cube):
        """
        Calculate the incidence rate measure and prepare the data for graphing
        """
        root_location = config.simulation_parameters.location_id
        pop = self.population_view.get(index)

        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}

        for sex in sexes:
            for location in locations:
                location_index = pop.query('location == @location').index if location >= 0 else pd.Index([])
                location_index = pop.index if location_index.empty else location_index
                incidence_rates = self.incidence_rate_df.loc[location_index]

                last_age_group_max = 0
                for age_bin, upr_bound in self.age_bin_age_group_max_dict:

                    event_count_column = '{}_event_count_{}_among_{}s'.format(self.disease, age_bin, sex)
                    succeptible_time_column = 'susceptible_person_time_{}_among_{}s'.format(age_bin, sex)

                    susceptible_person_time = incidence_rates[succeptible_time_column].sum()
                    num_cases = incidence_rates[event_count_column].sum()

                    if susceptible_person_time != 0:
                        cube = cube.append(pd.DataFrame({'measure': 'incidence',
                                                         'age_low': last_age_group_max,
                                                         'age_high': upr_bound,
                                                         'sex': sex,
                                                         'location': location if location >= 0 else root_location,
                                                         'cause': self.disease,
                                                         'value': num_cases/susceptible_person_time,
                                                         'sample_size': susceptible_person_time}, index=[0]).set_index(
                            ['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
                    last_age_group_max = upr_bound

        self.collecting = False

        for col in self.susceptible_person_time_cols + self.event_count_cols:
            self.incidence_rate_df[col] = 0

        return cube
