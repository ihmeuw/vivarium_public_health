import operator

import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config

from ceam_inputs import get_age_bins

from ceam_public_health.components.util import make_cols_demographically_specific

diarrhea_event_count_cols = make_cols_demographically_specific('diarrhea_event_count', 2, 5)
diarrhea_event_count_cols.append('diarrhea_event_count')


class IncidenceRateCalculation():
    # TODO: Need to figure out how pass in all of the diseases
    def __init__(self, disease_col, susceptible_col):
        self.disease_col = disease_col
        self.susceptible_col = susceptible_col

    def setup(self, builder):
        # get all gbd age groups
        age_bins = get_age_bins()

        # filter down all age groups to only the ones we care about
        # FIXME: the age groups of interest will change for GBD 2016, since the 85-90 age group is in GBD 2016, but not 2015
        age_bins = age_bins[(age_bins.age_group_id > 1) & (age_bins.age_group_id <= 5)]

        age_bins.age_group_name = age_bins.age_group_name.str.lower()

        age_bins.age_group_name = [x.strip().replace(' ', '_') for x in age_bins.age_group_name]

        self.dict_of_age_group_name_and_max_values = dict(zip(age_bins.age_group_name, age_bins.age_group_years_end))

    @listens_for('time_step', priority=9)
    @uses_columns(['diarrhea', 'age', 'sex'] + susceptible_person_time_cols + diarrhea_event_count_cols, 'alive')
    def count_time_steps_sim_has_diarrhea(self, event):
        pop = event.population_view.get(event.index)

        current_year = pd.Timestamp(event.time).year

        incidence_rate_df = pd.DataFrame()

        last_age_group_max = 0

        # sort self.dict_of_age_group_name_and_max_values by value (max age)
        self.sorted_dict = sorted(self.dict_of_age_group_name_and_max_values.items(), key=operator.itemgetter(1))

        for sex in ["Male", "Female"]:
            for key, value in self.sorted_dict:
                # FIXME: Do we want to calculate incidence rates on a per population or susceptible person time basis?
                # FIXME: Susceptible person time estimates are off unless end data falls exactly on a time step (so this is fine for the diarrhea model -- 1 day timesteps -- but may not be ok for other causes)
                pop.loc[(pop[self.disease_col] != self.susceptible_col) & (pop['age'] < value) & (pop['age'] >= last_age_group_max) & (pop['sex'] == sex), 'susceptible_person_time_{k}_in_year_{c}_among_{s}s'.format(k=key, c=current_year, s=sex)] += config.getfloat('simulation_parameters', 'time_step')
                num_diarrhea_cases = pop['diarrhea_event_count_{a}_in_year_{y}_among_{s}s'.format(a=key, y=year, s=sex)].sum()
                if susceptible_person_time != 0:
                    
                last_age_group_max = value
        return incidence_rate_dict
