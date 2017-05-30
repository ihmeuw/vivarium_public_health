import pandas as pd
import numpy as np
import os.path
import pdb

from functools import partial

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam.framework.randomness import choice

from ceam_public_health.util.risk import (natural_key, naturally_sort_df,
                                         assign_exposure_categories,
                                         assign_relative_risk_value)
from ceam_inputs import get_outpatient_visit_cost, get_ors_exposures, get_ors_relative_risks, \
                        get_ors_pafs


class ORS():
    """
    """
    def setup(self, builder):
        # pull PAFs to get risk-deleted mortality and relative risk to add mortality back in for those that get severe diarrhea but not ORS
        # FIXME: We could update the paf pipeline to be able to handle pafs that affect mortality (giving the self.paf variable a long name
        #     below to ensure it doesn't get put into the current paf pipeline)
        self.paf = builder.value('ors_population_attributable_fraction')
        self.paf.source = builder.lookup(get_ors_pafs())
        
        self.rr = builder.value('ors_relative_risk')
        self.rr.source = builder.lookup(get_ors_relative_risks())
        
        # pull exposure and include any interventions that change exposure
        ors_exposure = get_ors_exposures()
        
        if config.simulation_parameters.run_intervention:
            # add exposure above baseline increase in intervention scenario
            ors_exposure_increase_above_baseline = config.ORS.ors_exposure_increase_above_baseline
            ors_exposure['cat1'] -= ors_exposure_increase_above_baseline
            ors_exposure['cat2'] += ors_exposure_increase_above_baseline

        self.exposure = builder.value('exposure.ors')
        self.exposure.source = builder.lookup(ors_exposure)
        self.randomness = builder.randomness('ors_susceptibility')
        
        self.cost = get_outpatient_visit_cost()
        
        
    @listens_for('initialize_simulants')
    @uses_columns(['ors_count', 'ors_propensity', 'ors_outpatient_visit_cost', 'ors_working', 'ors_end_time', 'ors_outpatient_visit_cost', 'ors_facility_cost'])
    def load_columns(self, event):
        
        length = len(event.index)
        
        df = pd.DataFrame({'ors_count': [0]*length}, index=event.index)
        
        df['ors_propensity'] = pd.Series(self.randomness.get_draw(event.index), index=event.index)
        
        df['ors_end_time'] = pd.Series([pd.NaT]*length, index=event.index)
        
        df['ors_working'] = pd.Series([0]*length, index=event.index)
        
        df['ors_outpatient_visit_cost'] = pd.Series([0.0]*length, index=event.index)

        df['ors_facility_cost'] = pd.Series([0.0]*length, index=event.index)
       
        event.population_view.update(df)

    
    # TODO: Using a fake exposure and population of a bunch of people that just got diarrhea, check 
    @listens_for('time_step', priority=6)
    @uses_columns(['ors_propensity', 'diarrhea_event_time', 'diarrhea_event_end_time', 'ors_working', 'ors_end_time', 'ors_count', 'ors_outpatient_visit_cost'], 'alive')
    def determine_simulant_ORS_relative_risks(self, event):
        """
        This method determines who should be seeing the benefit of ORS
        """
        pop = event.population

        # if the simulant should no longer be receiving ORS, then set the working column to false
        pop.loc[pop['ors_end_time'] <= event.time, 'ors_working'] = 0
        
        # now we want to determine who should start receiving ORS this time step
        # filter down to only people that got diarrhea this time step
        # FIXME: people don't necessarily get diarrhea on the first day in which they get diarrhea. might want to inject some uncertainty here
        pop = pop.loc[pop['diarrhea_event_time'].notnull()]

        pop.loc[pop['diarrhea_event_time'] == pd.Timestamp(event.time)]

        exp = self.exposure(pop.index)
        
        exp, categories = naturally_sort_df(exp)
        
        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)

        exp = pop.join(exp)

        exp = assign_exposure_categories(exp, 'ors_propensity',
                                         categories)        
    
        pop = pop.join(exp)
        
        pop.loc[pop['exposure_category'] == 'cat2', 'ors_working'] = 1
        
        pop.loc[pop['ors_working'] == 1, 'ors_end_time'] = pop['diarrhea_event_end_time']
        pop.loc[pop['ors_working'] == 1, 'ors_count'] += 1
        
        current_year = pd.Timestamp(event.time).year
        current_cost = self.cost.query("year_id == {}".format(current_year)).set_index(['year_id']).loc[current_year]['cost']
        pop.loc[pop['ors_working'] == 1, 'ors_outpatient_visit_cost'] += current_cost
                
        event.population_view.update(pop)
    
    @modifies_value('excess_mortality.diarrhea')
    @uses_columns(['ors_working'], 'alive')
    def mortality_rates(self, index, rates, population_view):
        """
        
        """
        pop = population_view.get(index)

        # manually set the ORS-deleted mortality rate
        rates *= self.paf(index)
        
        # manually increase the diarrhea excess mortality rate for people that do not get ORS
        ors_not_working_index = pop.query("ors_working == 0").index

        # FIXME: Not sure if this is the best way to do things. Do I need the flatten here?
        rates.loc[ors_not_working_index] *= self.rr(ors_not_working_index)[['cat1']].values.flatten()
        
        return rates
        
    @modifies_value('metrics')
    @uses_columns(['ors_count', 'ors_outpatient_visit_cost', 'ors_facility_cost'])
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
            rotaviral_entiritis_vaccine_first_dose_count,
            rotaviral_entiritis_vaccine_second_dose_count,
            rotaviral_entiritis_vaccine_third_dose_count,
            rotaviral_entiritis_vaccine_unit_cost,
            cost_to_administer_rotaviral_entiritis_vaccine
        """
        population = population_view.get(index)

        metrics['ors_outpatient_visit_cost'] = population['ors_outpatient_visit_cost'].sum()
        metrics['ors_count'] = population['ors_count'].sum()
        metrics['ors_facility_cost'] = population['ors_count'].sum() * config.ORS.facility_cost

        return metrics
 
