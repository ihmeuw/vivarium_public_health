import operator

import pandas as pd, numpy as np

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config

from ceam_inputs import get_age_bins


class AccrueSusceptiblePersonTime():
    # TODO: Need to figure out how pass in all of the diseases
    def __init__(self, disease_col, susceptible_col):
        self.disease_col = disease_col
        self.susceptible_col = susceptible_col


    @listens_for('initialize_simulants')
    @uses_columns(['susceptible_person_time_early_neonatal', 'susceptible_person_time_late_neonatal', 'susceptible_person_time_post_neonatal', 'susceptible_person_time_1_to_4', 'susceptible_person_time_5_to_9', 'susceptible_person_time_10_to_14', 'susceptible_person_time_15_to_19', 'susceptible_person_time_20_to_24', 'susceptible_person_time_25_to_29', 'susceptible_person_time_30_to_34', 'susceptible_person_time_35_to_39', 'susceptible_person_time_40_to_44', 'susceptible_person_time_45_to_49', 'susceptible_person_time_50_to_54', 'susceptible_person_time_55_to_59', 'susceptible_person_time_60_to_64', 'susceptible_person_time_65_to_69', 'susceptible_person_time_70_to_74', 'susceptible_person_time_75_to_79', 'susceptible_person_time_80_plus'])
    def create_person_year_columns(self, event):
        length = len(event.index)
        event.population_view.update(pd.DataFrame({'susceptible_person_time_early_neonatal': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_late_neonatal': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_post_neonatal': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_1_to_4': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_5_to_9': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_10_to_14': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_15_to_19': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_20_to_24': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_25_to_29': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_30_to_34': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_35_to_39': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_40_to_44': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_45_to_49': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_50_to_54': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_55_to_59': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_60_to_64': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_65_to_69': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_70_to_74': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_75_to_79': np.zeros(length)}, index=event.index))
        event.population_view.update(pd.DataFrame({'susceptible_person_time_80_plus': np.zeros(length)}, index=event.index))


    def setup(self, builder):
        # get all gbd age groups
        age_bins = get_age_bins()

        # filter down all age groups to only the ones we care about
        # FIXME: the age groups of interest will change for GBD 2016, since the 85-90 age group is in GBD 2016, but not 2015
        age_bins = age_bins[(age_bins.age_group_id > 1) & (age_bins.age_group_id <= 21)]

        age_bins.age_group_name = age_bins.age_group_name.str.lower()

        age_bins.age_group_name = [x.strip().replace(' ', '_') for x in age_bins.age_group_name]

        self.dict_of_age_group_name_and_max_values = dict(zip(age_bins.age_group_name, age_bins.age_group_years_end))


    @listens_for('time_step', priority=9)
    @uses_columns(['diarrhea', 'susceptible_person_time_early_neonatal', 'susceptible_person_time_late_neonatal', 'susceptible_person_time_post_neonatal', 'susceptible_person_time_1_to_4', 'susceptible_person_time_5_to_9', 'susceptible_person_time_10_to_14', 'susceptible_person_time_15_to_19', 'susceptible_person_time_20_to_24', 'susceptible_person_time_25_to_29', 'susceptible_person_time_30_to_34', 'susceptible_person_time_35_to_39', 'susceptible_person_time_40_to_44', 'susceptible_person_time_45_to_49', 'susceptible_person_time_50_to_54', 'susceptible_person_time_55_to_59', 'susceptible_person_time_60_to_64', 'susceptible_person_time_65_to_69', 'susceptible_person_time_70_to_74', 'susceptible_person_time_75_to_79', 'susceptible_person_time_80_plus', 'age'], 'alive')
    def count_time_steps_sim_has_diarrhea(self, event):
        pop = event.population_view.get(event.index)

        last_age_group_max = 0

        # sort self.dict_of_age_group_name_and_max_values by value (max age)
        sorted_dict = sorted(self.dict_of_age_group_name_and_max_values.items(), key=operator.itemgetter(1))
        for key, value in sorted_dict:
            # FIXME: Susceptible person time estimates are off unless end data falls exactly on a time step (so this is fine for the diarrhea model -- 1 day timesteps -- but may not be ok for other causes)
            pop.loc[(pop[self.disease_col] != self.susceptible_col) & (pop['age'] < value) & (pop['age'] >= last_age_group_max), 'susceptible_person_time_{}'.format(key)] += config.getfloat('simulation_parameters', 'time_step')
            last_age_group_max = value


        event.population_view.update(pop)


    @modifies_value('metrics')
    @uses_columns(['susceptible_person_time_early_neonatal', 'susceptible_person_time_late_neonatal', 'susceptible_person_time_post_neonatal', 'susceptible_person_time_1_to_4', 'susceptible_person_time_5_to_9', 'susceptible_person_time_10_to_14', 'susceptible_person_time_15_to_19', 'susceptible_person_time_20_to_24', 'susceptible_person_time_25_to_29', 'susceptible_person_time_30_to_34', 'susceptible_person_time_35_to_39', 'susceptible_person_time_40_to_44', 'susceptible_person_time_45_to_49', 'susceptible_person_time_50_to_54', 'susceptible_person_time_55_to_59', 'susceptible_person_time_60_to_64', 'susceptible_person_time_65_to_69', 'susceptible_person_time_70_to_74', 'susceptible_person_time_75_to_79', 'susceptible_person_time_80_plus'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)

        metrics['susceptible_person_time_early_neonatal'] = population['susceptible_person_time_early_neonatal'].sum()
        metrics['susceptible_person_time_late_neonatal'] = population['susceptible_person_time_late_neonatal'].sum()
        metrics['susceptible_person_time_post_neonatal'] = population['susceptible_person_time_late_neonatal'].sum()
        metrics['susceptible_person_time_1_to_4'] = population['susceptible_person_time_1_to_4'].sum()
        metrics['susceptible_person_time_5_to_9'] = population['susceptible_person_time_5_to_9'].sum()
        metrics['susceptible_person_time_10_to_14'] = population['susceptible_person_time_10_to_14'].sum()
        metrics['susceptible_person_time_15_to_19'] = population['susceptible_person_time_15_to_19'].sum()
        metrics['susceptible_person_time_20_to_24'] = population['susceptible_person_time_20_to_24'].sum()
        metrics['susceptible_person_time_25_to_29'] = population['susceptible_person_time_25_to_29'].sum()
        metrics['susceptible_person_time_30_to_34'] = population['susceptible_person_time_30_to_34'].sum()
        metrics['susceptible_person_time_35_to_39'] = population['susceptible_person_time_35_to_39'].sum()
        metrics['susceptible_person_time_40_to_44'] = population['susceptible_person_time_40_to_44'].sum()
        metrics['susceptible_person_time_45_to_49'] = population['susceptible_person_time_45_to_49'].sum()
        metrics['susceptible_person_time_50_to_54'] = population['susceptible_person_time_50_to_54'].sum()
        metrics['susceptible_person_time_55_to_59'] = population['susceptible_person_time_55_to_59'].sum()
        metrics['susceptible_person_time_60_to_64'] = population['susceptible_person_time_60_to_64'].sum()
        metrics['susceptible_person_time_65_to_69'] = population['susceptible_person_time_65_to_69'].sum()
        metrics['susceptible_person_time_70_to_74'] = population['susceptible_person_time_70_to_74'].sum()
        metrics['susceptible_person_time_75_to_79'] = population['susceptible_person_time_75_to_79'].sum()
        metrics['susceptible_person_time_80_plus'] = population['susceptible_person_time_80_plus'].sum()


        return metrics


# End.
