import pandas as pd
import numpy as np

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs import (get_outpatient_visit_costs, get_ors_exposures,
                         get_ors_relative_risks, get_ors_pafs)


class Ors:
    """
    The Ors class accomplishes several things
    1) Reads in all ors risk data (pafs, relative risks, and exposures) and
        outpatient visit costs (we are setting the unit cost of ors to be the
        cost of an outpatient visit)
        #FIXME: Should was also include unit cost estimate for ors? Certainly
        some locations wouldn't have the ors cost baked into the visit cost
    2) If config.ors.run_intervention is set to True, the exposure will be
        updated based on the value in
        config.ors.ors_exposure_increase_above_baseline
    3) Creates the columns necessary to the component
    4) Determines which simulants are currently receiving ors
    5) Sets the lack of ors-deleted excess mortality rate for all simulants. For
        simulants that do not receive ors, we multiply the lack of ors-deleted
        mortality rate by the relative risk
    6) Outputs metrics for ors costs and counts
    """
    def setup(self, builder):
        # FIXME: We could update the paf and relative risk pipelines to be able
        #   to handle pafs that affect mortality (giving the self.paf variable
        #   a long name below to ensure it doesn't get put into the current paf
        #   pipeline)
        self.paf = builder.value('ors_population_attributable_fraction')
        self.paf.source = builder.lookup(get_ors_pafs())

        self.rr = builder.value('ors_relative_risk')
        self.rr.source = builder.lookup(get_ors_relative_risks())

        # pull exposure and include any interventions that change exposure
        ors_exposure = get_ors_exposures()

        if config.ors.run_intervention:
            # add exposure above baseline increase in intervention scenario
            exposure_increase = config.ors.ors_exposure_increase_above_baseline
            ors_exposure['cat1'] -= exposure_increase
            ors_exposure['cat2'] += exposure_increase

        self.exposure = builder.value('exposure.ors')
        self.exposure.source = builder.lookup(ors_exposure)
        self.randomness = builder.randomness('ors_susceptibility')

        self.cost = get_outpatient_visit_costs()

    @listens_for('initialize_simulants')
    @uses_columns(['ors_count', 'ors_propensity', 'ors_outpatient_visit_cost',
                   'ors_working', 'ors_end_time', 'ors_outpatient_visit_cost',
                   'ors_facility_cost'])
    def load_columns(self, event):
        """
        Creates count, propensity, working, and cost columns
        """
        length = len(event.index)
        df = pd.DataFrame({'ors_count': [0]*length,
                           'ors_propensity': self.randomness.get_draw(event.index),
                           'ors_end_time': [pd.NaT]*length,
                           'ors_working': [0]*length,
                           'ors_outpatient_visit_cost': [0.0]*length,
                           'ors_facility_cost': [0.0]*length}, index=event.index)
        event.population_view.update(df)

    @listens_for('time_step', priority=7)
    @uses_columns(['ors_propensity', 'diarrhea_event_time',
                   'diarrhea_event_end_time', 'ors_working', 'ors_end_time',
                   'ors_count', 'ors_outpatient_visit_cost'], 'alive')
    def determine_who_gets_ors(self, event):
        """
        This method determines who should be seeing the benefit of ors
        """
        pop = event.population

        # if the simulant should no longer be receiving ors, then set the
        # working column to false
        pop.loc[pop['ors_end_time'] <= event.time, 'ors_working'] = 0

        # now we want to determine who should start receiving ors this time
        # step filter down to only people that got diarrhea this time step
        # start by filtering out people that have never had diarrhea (the next
        # line will give us people that have never had a case if we don't
        # filter out people that have never had diarrhea first)
        pop = pop.loc[pop['diarrhea_event_time'].notnull()]

        # FIXME: people don't necessarily get diarrhea on the first day in
        # which they get diarrhea. might want to inject some uncertainty here
        pop = pop.loc[pop['diarrhea_event_time'] == pd.Timestamp(event.time)]

        exp = self.exposure(pop.index)

        categories = sorted([c for c in exp.columns if 'cat' in c],
                            key=lambda c: int(c.split('cat')[1]))
        exp = exp[categories]
        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)
        exp = pop.join(exp)
        exp = assign_exposure_categories(exp, 'ors_propensity', categories)

        pop = pop.join(exp)

        pop.loc[pop['exposure_category'] == 'cat2', 'ors_working'] = 1

        pop.loc[pop['ors_working'] == 1, 'ors_end_time'] = pop['diarrhea_event_end_time']
        pop.loc[pop['ors_working'] == 1, 'ors_count'] += 1

        # outpatient visit costs vary by year within a location. get the cost
        # for the current year
        current_year = pd.Timestamp(event.time).year
        current_cost = self.cost.query("year_id == {}".format(
            current_year)).set_index(['year_id']).loc[current_year]['cost']
        pop.loc[pop['ors_working'] == 1, 'ors_outpatient_visit_cost'] += \
            current_cost

        event.population_view.update(pop)

    # FIXME: Need to ensure the mortality rates calculation happens after
    #     determine_who_gets_ors
    @modifies_value('excess_mortality.diarrhea')
    @uses_columns(['ors_working'])
    def mortality_rates(self, index, rates, population_view):
        """
        Set the lack of ors-deleted mortality rate for all simulants. For those
        exposed to the risk (the risk is the ABSENCE of ors), multiply the
        lack of ors-deleted excess mortality rate by the relative risk
        """
        pop = population_view.get(index)

        # manually set the lack of ors-deleted mortality rate
        rates *= 1 - self.paf(index)

        # manually increase the diarrhea excess mortality rate for people that
        #     do not get ors (i.e. those exposed to the lack of ors risk)
        ors_not_working_index = pop.query("ors_working == 0").index

        if not ors_not_working_index.empty:
            # FIXME: Not sure if this is the best way to do things.
            #     Do I need the flatten here?
            rates.loc[ors_not_working_index] *= self.rr(ors_not_working_index)[['cat1']].values.flatten()

        return rates

    @modifies_value('metrics')
    @uses_columns(['ors_count', 'ors_outpatient_visit_cost',
                   'ors_facility_cost'])
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
            'ors_count', 'ors_outpatient_visit_cost', and 'ors_facility_cost'
        """
        population = population_view.get(index)

        metrics['ors_outpatient_visit_cost'] = population['ors_outpatient_visit_cost'].sum()
        metrics['ors_count'] = population['ors_count'].sum()
        metrics['ors_facility_cost'] = population['ors_count'].sum() * config.ors.facility_cost

        return metrics


def assign_exposure_categories(df, susceptibility_column, categories):
    """Creates an 'exposure_category' column that assigns 
    simulant's exposure based on their susceptibility draw

    Parameters
    ----------
    df : pd.DataFrame
    susceptibility_column : str
    categories : list
        list of all of the category columns in df 
    """

    bool_list = [c + '_bool' for c in categories]

    # TODO: Confirm whether or not we want < or <=
    for col in categories:
        df['{}_bool'.format(col)] = df['{}'.format(col)] < df[susceptibility_column]

    df['exposure_category'] = df[bool_list].sum(axis=1)

    # seems weird, but we need to add 1 to exposure category. e.g. if all values for a row in bool_list are false
    # that simulant will be in exposure cat1, not cat0
    df['exposure_category'] = df['exposure_category'] + 1

    df['exposure_category'] = 'cat' + df['exposure_category'].astype(str)

    return df[['exposure_category']]


def assign_relative_risk_value(df, categories):
    """
    creates an 'relative_risk_value' column that assigns simulant's relative risk based on their exposure

    Parameters
    ----------
    df : pd.DataFrame

    susceptibility_column : pd.Series

    categories : list
        list of all of the category columns in df

    """

    for col in categories:
        df.loc[(df['exposure_category'] == col, 'relative_risk_value')] = df[col]

    return df
