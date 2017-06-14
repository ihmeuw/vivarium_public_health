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
        # FIXME: Get rid of susceptible state. Instead want to have a list of diseased states
        self.disease_states = disease_states
        # FIXME: Update so all age groups are included. Do this after updating code to not affect state table
        self.age_group_id_min = 2
        self.age_group_id_max = 6
        self.collecting = False

        self.susceptible_person_time_cols = make_cols_demographically_specific("susceptible_person_time", self.age_group_id_min, self.age_group_id_max)
        self.event_count_cols = make_cols_demographically_specific("{}_event_count".format(self.disease), self.age_group_id_min, self.age_group_id_max)

        self.age_bin_age_group_max_dict = make_age_bin_age_group_max_dict(age_group_id_min=self.age_group_id_min,
                                                                          age_group_id_max=self.age_group_id_max)

    def setup(self, builder):
        self.clock = builder.clock()

        columns = [self.disease_col, "age", "sex", "alive"] + self.susceptible_person_time_cols + self.event_count_cols
        self.population_view = builder.population_view(columns)

    @listens_for('initialize_simulants')
    def create_columns(self, event):
        """
        Initialize the susceptible person time and event count columns
        """
        length = len(event.index)

        for col in self.susceptible_person_time_cols:
            self.population_view.update(pd.DataFrame({col: np.zeros(length)}, index=event.index))

        for col in self.event_count_cols:
            self.population_view.update(pd.DataFrame({col: np.zeros(length)}, index=event.index))

    @listens_for('begin_epidemiological_measure_collection')
    def set_flag(self, event):
        """
        Set the collecting flag to True during GBD years
        """
        # FIXME: Figure out how to turn off the self.collecting flag
        self.collecting = True

    @listens_for('time_step', priority=7)
    def get_counts_and_susceptible_person_time(self, event):
        """
        Gather all of the data we need for the incidence rate calculations (event counts and susceptible person time)
        """
        if self.collecting:
            pop = self.population_view.get(event.index)

            current_year = event.time.year

            # FIXME: Move away from collecting incidence rate data in the state table

            for sex in ["Male", "Female"]:
                last_age_group_max = 0
                for age_bin, upr_bound in self.age_bin_age_group_max_dict:
                    # We use GTE age group lower bound and LT age group upper bound
                    #     because of how GBD age groups are set up. For example, a
                    #     A simulant can be 1 or 4.999 years old and be considered
                    #     part of the 1-5 year old group, but once they turn 5 they
                    #     are part of the 5-10 age group
                    pop.loc[(pop['age'] < upr_bound)
                            & (pop['age'] >= last_age_group_max)
                            & (pop['sex'] == sex)
                            & (pop[self.disease_col].isin(self.disease_states))
                            & (pop['alive'] == True),
                            '{d}_event_count_{a}_in_year_{c}_among_{s}s'.format(
                            d=self.disease, a=age_bin, c=current_year, s=sex)] += 1
                    pop.loc[~(pop[self.disease_col].isin(self.disease_states))
                            & (pop['age'] < upr_bound)
                            & (pop['age'] >= last_age_group_max)
                            & (pop['sex'] == sex)
                            & (pop['alive'] == True),
                            'susceptible_person_time_{a}_in_year_{c}_among_{s}s'.format(a=age_bin, c=current_year, s=sex)] += config.simulation_parameters.time_step
                    last_age_group_max = upr_bound

            self.population_view.update(pop)

    # TODO: Would be nice to use age_group_name instead of age_group_high and age_group_low. Using age_group_name is more specific, will make the graphs cleaner, and is more interpretable for the under 1 (neonatal) age groups.
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

        now = self.clock()
        window_start = now - duration
        current_year = window_start.year

        for sex in sexes:
            for location in locations:
                last_age_group_max = 0
                for age_bin, upr_bound in self.age_bin_age_group_max_dict:
                    # FIXME: We want to make sure our mortality rates and prevalences and incidence rates are all using >= lower and < upper. Don't delete this FIXME until prevalence and mortality are made to be similar
                    if location >= 0:
                        pop = pop.query('location == @location')

                    susceptible_person_time = pop["susceptible_person_time_{a}_in_year_{y}_among_{s}s".format(a=age_bin, y=current_year, s=sex)].sum()
                    num_diarrhea_cases = pop['{d}_event_count_{a}_in_year_{y}_among_{s}s'.format(d=self.disease, a=age_bin, y=current_year, s=sex)].sum()

                    if susceptible_person_time != 0:
                        cube = cube.append(pd.DataFrame({'measure': 'incidence', 'age_low': last_age_group_max, 'age_high': upr_bound, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': 'diarrhea', 'value': num_diarrhea_cases/susceptible_person_time, 'sample_size': susceptible_person_time}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))

                    last_age_group_max = upr_bound

        self.collecting = False

        return cube
