import operator

import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config

from ceam_inputs import get_age_bins


# TODO: Don't duplicate code! Get rid of the duplicate lines in the block below and setup
# get all gbd age groups
age_bins = get_age_bins()

# filter down all age groups to only the ones we care about
# FIXME: the age groups of interest will change for GBD 2016, since the 85-90 age group is in GBD 2016, but not 2015
age_bins = age_bins[(age_bins.age_group_id > 1) & (age_bins.age_group_id <= 5)]

age_bins.age_group_name = age_bins.age_group_name.str.lower()

age_bins.age_group_name = [x.strip().replace(' ', '_') for x in age_bins.age_group_name]

list_of_age_bins = []
susceptible_person_time_cols = []
diarrhea_event_count_cols = []

year_start = config.getint('simulation_parameters', 'year_start')
year_end = config.getint('simulation_parameters', 'year_end')

for age_bin in pd.unique(age_bins.age_group_name.values):
    list_of_age_bins.append(age_bin)
    for year in range(year_start, year_end+1):
        for sex in ['Male', 'Female']:
            susceptible_person_time_cols.append("susceptible_person_time_" + age_bin + "_in_year_{}".format(year) + "_among_" + sex + "s")
            diarrhea_event_count_cols.append('diarrhea_event_count_{a}_in_year_{y}_among_{s}s'.format(a=age_bin, y=year, s=sex))
            diarrhea_event_count_cols.append('diarrhea_event_count')


class AccrueSusceptiblePersonTime():
    # TODO: Need to figure out how pass in all of the diseases
    def __init__(self, disease_col, susceptible_col):
        self.disease_col = disease_col
        self.susceptible_col = susceptible_col


    @listens_for('initialize_simulants')
    @uses_columns(susceptible_person_time_cols)
    def create_person_year_columns(self, event):
        length = len(event.index)
        for col in susceptible_person_time_cols:
            event.population_view.update(pd.DataFrame({col: np.zeros(length)}, index=event.index))


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
    @uses_columns(['diarrhea', 'age', 'sex'] + susceptible_person_time_cols, 'alive')
    def count_time_steps_sim_has_diarrhea(self, event):
        pop = event.population_view.get(event.index)

        current_year = pd.Timestamp(event.time).year

        last_age_group_max = 0

        # sort self.dict_of_age_group_name_and_max_values by value (max age)
        self.sorted_dict = sorted(self.dict_of_age_group_name_and_max_values.items(), key=operator.itemgetter(1))

        for sex in ["Male", "Female"]:
            for key, value in self.sorted_dict:
                # FIXME: Susceptible person time estimates are off unless end data falls exactly on a time step (so this is fine for the diarrhea model -- 1 day timesteps -- but may not be ok for other causes)
                pop.loc[(pop[self.disease_col] != self.susceptible_col) & (pop['age'] < value) & (pop['age'] >= last_age_group_max) & (pop['sex'] == sex), 'susceptible_person_time_{k}_in_year_{c}_among_{s}s'.format(k=key, c=current_year, s=sex)] += config.getfloat('simulation_parameters', 'time_step')
                last_age_group_max = value

        event.population_view.update(pop)


    @modifies_value('metrics')
    @uses_columns(['cause_of_death', 'death_day'] + susceptible_person_time_cols)
    def calculate_mortality_rates(self, index, metrics, population_view):
        # if config.getint('simulation_parameters', 'epi_analysis') == 0:
        #    pass

        # if config.getint('simulation_parameters', 'epi_analysis') == 1:

        pop = population_view.get(index)

        mortality_df = pd.DataFrame(columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value', 'year', 'draw'])

        last_age_group_max = 0

        pop['death_year'] = pop['death_day'].map(lambda x: x.year)

        for key, value in self.sorted_dict:
            for year in range(year_start, year_end+1):
                for sex in ['Male', 'Female']:
                    susceptible_person_time = pop["susceptible_person_time_{a}_in_year_{y}_among_{s}s".format(a=key, y=year, s=sex)].sum()
                    deaths_due_to_diarrhea = len(pop.query("death_year == {} and cause_of_death=='death_due_to_severe_diarrhea'".format(year)))
                    if susceptible_person_time != 0:
                        metrics["mortality_rate_" + key + "_in_year_{}".format(year) + "_among_" + sex + "s"] = deaths_due_to_diarrhea / susceptible_person_time
                        row = pd.DataFrame({'measure': ['mortality'], 'age_low': [last_age_group_max], 'age_high': [value], 'sex': [sex], 'location': [180], 'cause': ['diarrhea'], 'value': [deaths_due_to_diarrhea / susceptible_person_time], 'year': [year], 'draw': [0]})
                        mortality_df = mortality_df.append(row)
            last_age_group_max = value

        # mortality_df.to_hdf("/share/scratch/users/emumford/mortality_rate.hdf", key="key")

        return metrics


    @modifies_value('metrics')
    @uses_columns(['cause_of_death', 'death_day'] + susceptible_person_time_cols + diarrhea_event_count_cols)
    def calculate_incidence_rates(self, index, metrics, population_view):
        # if config.getint('simulation_parameters', 'epi_analysis') == 0:
        #    pass

        # if config.getint('simulation_parameters', 'epi_analysis') == 1:

        pop = population_view.get(index)

        incidence_df = pd.DataFrame(columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value', 'year', 'draw'])

        last_age_group_max = 0

        for key, value in self.sorted_dict:
            for year in range(year_start, year_end+1):
                for sex in ['Male', 'Female']:
                    susceptible_person_time = pop["susceptible_person_time_{a}_in_year_{y}_among_{s}s".format(a=key, y=year, s=sex)].sum()
                    num_diarrhea_cases = pop['diarrhea_event_count_{a}_in_year_{y}_among_{s}s'.format(a=key, y=year, s=sex)].sum()
                    if susceptible_person_time != 0:
                        metrics["incidence_rate_" + key + "_in_year_{}".format(year) + "_among_" + sex + "s"] = num_diarrhea_cases / susceptible_person_time
                        row = pd.DataFrame({'measure': ['incidence'], 'age_low': [last_age_group_max], 'age_high': [value], 'sex': [sex], 'location': [180], 'cause': ['diarrhea'], 'value': [num_diarrhea_cases / susceptible_person_time], 'year': [year], 'draw': [0]})
                        incidence_df = incidence_df.append(row)
            last_age_group_max = value

        # incidence_df = incidence_df.to_hdf("/share/scratch/users/emumford/incidence_rate.hdf", key="key")

        return metrics


    # @modifies_value('metrics')
    # @uses_columns(susceptible_person_time_cols)
    # def metrics(self, index, metrics, population_view):
    #    population = population_view.get(index)

    #    for col in susceptible_person_time_cols:
    #        metrics[col] = population[col].sum()

    #    return metrics


# End.
