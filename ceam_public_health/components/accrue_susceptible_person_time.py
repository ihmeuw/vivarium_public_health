import pandas as pd, numpy as np
from db_tools import ezfuncs
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

class AccrueSusceptiblePersonTime():
    # TODO: Need to figure out how pass in all of the diseases
    def __init__(self, disease_col, susceptible_col):
        self.disease_col = disease_col
        self.susceptible_col = susceptible_col


    @listens_for('initialize_simulants')
    @uses_columns(['simulant_initialization_time', 'susceptible_person_time_under_5', 'susceptible_person_time_over_5'])
    def create_person_year_columns(event):
        length = len(event.index)
        event.population_view.update(pd.DataFrame({'simulant_initialization_time': [pd.Timestamp(event.time)]*length}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_early_neonatal': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_late_neonatal': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_post_neonatal': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_1_to_4': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_5_to_9': np.zeros(length)}, index=event.index))

    def setup(self, builder):
        # get all gbd age groups
        age_bins = ezfuncs.query('''select age_group_id, age_group_years_start, age_group_years_end, age_group_name from age_group''', conn_def='shared')

        # filter down all age groups to only the ones we care about
        # FIXME: the age groups of interest will change for GBD 2016, since the 85-90 age group is in GBD 2016, but not 2015
        age_bins = age_bins[(age_bins.age_group_id > 1) & (age_bins.age_group_id <= 21)]

        age_bins.age_group_name = age_bins.age_group_name.str.lower()

        age_bins.age_group_name = [x.strip().replace(' ', '_') for x in age_bins.age_group_name]

        self.dict_of_age_group_name_and_max_values = dict(zip(age_bins.age_group_name, age_bins.age_group_years_end))


    @listens_for('time_step', priority=9)
    @uses_columns(['diarrhea', 'susceptible_person_time_early_neonatal', 'susceptible_person_time_late_neonatal', 'susceptible_person_time_post_neonatal', 'age'], 'alive')
    def count_time_steps_sim_has_diarrhea(event):
        pop = event.population_view.get(event.index)

        last_age_group_max = 0

        # TODO: Make sure the susceptible person time accrual is working, especially for under 1 year olds
        for key, value in self.dict_of_age_group_name_and_max_values:
            # FIXME: Susceptible person time estimates are off unless end data falls exactly on a time step (so this is fine for the diarrhea model -- 1 day timesteps -- but may not be ok for other causes)
            pop.loc[(pop[self.disease_col] != self.susceptible_col) & (pop['age'] < value) & (pop['age'] >= last_age_group_max), 'susceptible_person_time_{}'.format()] += config.getint('simulation_parameters', 'time_step')
            last_age_group_max = value

            import pdb; pdb.set_trace()

        event.population_view.update(pop)

