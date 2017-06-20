import operator

import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config

from ceam_inputs import get_age_bins

from ceam_public_health.util import make_cols_demographically_specific, make_age_bin_age_group_max_dict, make_age_bin_age_group_max_dict


class CalculateIncidence:
    def __init__(self, disease_col, disease, disease_states):
        """
        disease_col: str
            name of the column name that contains the disease state of interest

        disease: str
            name of the disease of interest

        disease_states: list
            list of states that denote a simulant as having the disease (e.g. ['severe_diarrhea', 'moderate_diarrhea', 'mild_diarrhea']). If a simulant does not have the disease of interest, we say that they are in the susceptible state
        """
        self.disease_col = disease_col
        self.disease = disease
        self.disease_time_col = disease + "_event_time"
        self.disease_states = disease_states
        self.collecting = False

        self.susceptible_person_time_cols = make_cols_demographically_specific("susceptible_person_time", age_group_id_min=2, age_group_id_max=21)
        self.event_count_cols = make_cols_demographically_specific("{}_event_count".format(self.disease), age_group_id_min=2, age_group_id_max=21)

        self.age_bin_age_group_max_dict = make_age_bin_age_group_max_dict(age_group_id_min=2,
                                                                          age_group_id_max=21)

    def setup(self, builder):
        self.clock = builder.clock()

        columns = [self.disease_col, self.disease_time_col, "age", "sex", "alive"]
        self.population_view = builder.population_view(columns)

    @listens_for('begin_epidemiological_measure_collection')
    def set_flag(self, event):
        """
        Set the collecting flag to True during GBD years
        """
        # FIXME: Figure out how to turn off the self.collecting flag
        self.collecting = True

        self.incidence_rate_df = pd.DataFrame({})

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
            pop = self.population_view.get(event.index)

            for sex in ["Male", "Female"]:
                last_age_group_max = 0
                for age_bin, upr_bound in self.age_bin_age_group_max_dict:
                    # We use GTE age group lower bound and LT age group upper bound
                    #     because of how GBD age groups are set up. For example, a
                    #     A simulant can be 1 or 4.999 years old and be considered
                    #     part of the 1-4 year old group, but once they turn 5 they
                    #     are part of the 5-10 age group
                    cases_index = pop.loc[(pop['age'] < upr_bound)
                                & (pop['age'] >= last_age_group_max)
                                & (pop['sex'] == sex)
                                & (pop[self.disease_col].isin(self.disease_states))
                                & (pop['alive'] == True)
                                & (pop[self.disease_time_col] == event.time)].index
                    self.incidence_rate_df['{d}_event_count_{a}_among_{s}s'.format(
                            d=self.disease, a=age_bin, s=sex)].loc[cases_index] += 1
                    susceptible_index = pop.loc[~(pop[self.disease_col].isin(self.disease_states))
                                              & (pop['age'] < upr_bound)
                                              & (pop['age'] >= last_age_group_max)
                                              & (pop['sex'] == sex)
                                              & (pop['alive'] == True)].index
                    # calculate susceptible person-time per year
                    self.incidence_rate_df['susceptible_person_time_{a}_among_{s}s'.format(a=age_bin, s=sex)].loc[susceptible_index] += config.simulation_parameters.time_step / 365
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
                last_age_group_max = 0
                for age_bin, upr_bound in self.age_bin_age_group_max_dict:
                    location_index = pd.Index([])
                    if location >= 0:
                        location_index = pop.query('location == @location').index
                    if location_index.empty:
                        location_index = pop.index
                    susceptible_person_time = self.incidence_rate_df.loc[location_index]["susceptible_person_time_{a}_among_{s}s".format(a=age_bin, s=sex)].sum()
                    num_cases = self.incidence_rate_df.loc[location_index]['{d}_event_count_{a}_among_{s}s'.format(d=self.disease, a=age_bin, s=sex)].sum()

                    if susceptible_person_time != 0:
                        cube = cube.append(pd.DataFrame({'measure': 'incidence', 'age_low': last_age_group_max, 'age_high': upr_bound, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': self.disease, 'value': num_cases/susceptible_person_time, 'sample_size': susceptible_person_time}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
                    last_age_group_max = upr_bound

        self.collecting = False

        for col in self.susceptible_person_time_cols:
            self.incidence_rate_df[col] = 0

        for col in self.event_count_cols:
            self.incidence_rate_df[col] = 0
           
        return cube
