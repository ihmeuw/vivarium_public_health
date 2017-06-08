import operator

import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config

from ceam_public_health.components.util import make_age_bin_age_group_max_dict

from ceam_inputs import get_age_bins

from ceam_public_health.components.util import make_cols_demographically_specific, make_age_bin_age_group_max_dict

SUSCEPTIBLE_PERSON_TIME_COLS = make_cols_demographically_specific("susceptible_person_time", 2, 5)
DIARRHEA_EVENT_COUNT_COLS = make_cols_demographically_specific('diarrhea_event_count', 2, 5)
DIARRHEA_EVENT_COUNT_COLS.append('diarrhea_event_count')

class CalculateIncidence:
    def __init__(self, disease_col, disease, susceptible_state, age_group_id_min, age_group_id_max):
        """
         
        """
        self.disease_col = disease_col
        self.susceptible_state = susceptible_state
        self.age_group_id_min = age_group_id_min
        self.age_group_id_max = age_group_id_max
        self.disease = disease
        self.collecting = False

    @listens_for('initialize_simulants')
    @uses_columns(SUSCEPTIBLE_PERSON_TIME_COLS)
    def create_person_year_columns(self, event):
        length = len(event.index)
        for col in SUSCEPTIBLE_PERSON_TIME_COLS:
            event.population_view.update(pd.DataFrame({col: np.zeros(length)}, index=event.index))

    @listens_for('begin_epidemiological_measure_collection')
    def set_flag(self, event):
        self.collecting = True

    @listens_for('time_step', priority=7)
    @uses_columns(['diarrhea', 'age', 'sex'] + DIARRHEA_EVENT_COUNT_COLS, 'alive')
    def get_counts_and_susceptible_person_time(self, event):
        if self.collecting:

            pop = event.population

            age_bin_age_group_max_dict = make_age_bin_age_group_max_dict(age_group_id_min=self.age_group_id_min,
                                                                         age_group_id_max=self.age_group_id_max)

            for sex in ["Male", "Female"]:
                last_age_group_max = 0
                for age_bin, upr_bound in age_bin_age_group_max_dict:
                    # We use GTE age group lower bound and LT age group upper bound
                    #     because of how GBD age groups are set up. For example, a
                    #     A simulant can be 1 or 4.999 years old and be considered
                    #     part of the 1-5 year old group, but once they turn 5 they
                    #     are part of the 5-10 age group
                    pop.loc[(pop['age'] < upr_bound)
                            & (pop['age'] >= last_age_group_max)
                            & (pop['sex'] == sex)
                            & (pop[self.disease_col] != self.susceptible_state),
                            '{d}_event_count_{a}_in_year_{c}_among_{s}s'.format(
                            d=self.disease, a=age_bin, c=current_year, s=sex)] += 1
                    # FIXME: Set up line below so that it uses isin. We want to
                    # check that self.disease_col is not in some range of diseased
                    # values (e.g. severe diarrhea, moderate diarrhea, etc)
                    pop.loc[(pop[self.disease_col] == self.susceptible_state)
                            & (pop['age'] < upr_bound)
                            & (pop['age'] >= last_age_group_max)
                            & (pop['sex'] == sex),
                            'susceptible_person_time_{k}_in_year_{c}_among_{s}s'.format(k=key, c=current_year, s=sex)] += config.simulation_parameters.time_step
                    last_age_group_max = upr_bound

            event.population_view.update(pop)

    # TODO: Would be nice to use age_group_name instead of age_group_high and age_group_low. Using age_group_name is more specific, will make the graphs cleaner, and is more interpretable for the under 1 (neonatal) age groups.
    # FIXME: Should move the epi measures code to its own class, probably its own script
    @modifies_value('epidemiological_span_measures')
    @uses_columns(['age', 'death_day', 'cause_of_death', 'alive', 'sex'] + SUSCEPTIBLE_PERSON_TIME_COLS + DIARRHEA_EVENT_COUNT_COLS)
    def calculate_incidence_measure(self, index, age_groups, sexes, all_locations, duration, cube, population_view):
        root_location = config.getint('simulation_parameters', 'location_id')
        pop = population_view.get(index)

        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}

        now = self.clock()
        window_start = now - duration
        current_year = window_start.year


        # FIXME: Don't want to have age_groups[0:3] hard-coded in. Need to make a component that calculates susceptible person time for all age groups so that this can be avoided
        for low, high in age_groups[0:4]:
            for sex in sexes:
                for location in locations:
                    sub_pop = pop.query('age > @low and age <= @high and sex == @sex')
                    low_str = str(np.round(low, 2))
                    high_str = str(np.round(high, 2))
                    if location >= 0:
                        sub_pop = sub_pop.query('location == @location')

                    # TODO: Make this more flexible. Don't want to have diarrhea hard-coded in here. Want the susceptibility column and disease column to be variables that get passed into the class.
                    # TODO: Need to figure out best place for this
                    if not sub_pop.empty:
                        susceptible_person_time = pop["susceptible_person_time_{l}_to_{h}_in_year_{y}_among_{s}s".format(l=low_str, h=high_str, y=current_year, s=sex)].sum()
                        num_diarrhea_cases = pop['diarrhea_event_count_{l}_to_{h}_in_year_{y}_among_{s}s'.format(l=low_str, h=high_str, y=current_year, s=sex)].sum()
                        if susceptible_person_time != 0:
                            cube = cube.append(pd.DataFrame({'measure': 'incidence', 'age_low': low, 'age_high': high, 'sex': sex, 'location': location if location >= 0 else root_location, 'cause': 'diarrhea', 'value': num_diarrhea_cases/susceptible_person_time, 'sample_size': len(sub_pop)}, index=[0]).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube


# after dumping results, set the flag to false



#    @modifies_value('metrics')
#    @uses_columns(['cause_of_death', 'death_day'] + SUSCEPTIBLE_PERSON_TIME_COLS + DIARRHEA_EVENT_COUNT_COLS)
#    def calculate_incidence_rates(self, index, metrics, population_view):
#        pop = population_view.get(index)

#        incidence_df = pd.DataFrame(columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value', 'year', 'draw'])

#        last_age_group_max = 0

#        for key, value in self.sorted_dict:
#            for year in range(year_start, year_end+1):
#                for sex in ['Male', 'Female']:
#                    susceptible_person_time = pop["susceptible_person_time_{a}_in_year_{y}_among_{s}s".format(a=key, y=year, s=sex)].sum()
#                    num_diarrhea_cases = pop['diarrhea_event_count_{a}_in_year_{y}_among_{s}s'.format(a=key, y=year, s=sex)].sum()
#                    if susceptible_person_time != 0:
#                        metrics["incidence_rate_" + key + "_in_year_{}".format(year) + "_among_" + sex + "s"] = num_diarrhea_cases / susceptible_person_time
#                        row = pd.DataFrame({'measure': ['incidence'], 'age_low': [last_age_group_max], 'age_high': [value], 'sex': [sex], 'location': [180], 'cause': ['diarrhea'], 'value': [num_diarrhea_cases / susceptible_person_time], 'year': [year], 'draw': [0]})
#                        incidence_df = incidence_df.append(row)
#            last_age_group_max = value

        # incidence_df = incidence_df.to_hdf("/share/scratch/users/emumford/incidence_rate.hdf", key="key")
