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

from ceam_public_health.util.risk import natural_key, naturally_sort_df, \
                                         assign_exposure_categories, \
                                         assign_relative_risk_value
from ceam_inputs import get_exposures, get_relative_risks, \
                        get_pafs, get_ors_exposure


# TODO: Get rid of duplication in below + ceam_public_health/util/risk.py.

# TODO: Incorporate PAFs -- Is there even a PAF so that we can get ORS-deleted
#     incidence?

# FIXME: We're not actually ensuring that the correct proportion of simulants
#     are receiving ORS. We would need to take a draw each timestep to do that
#     which we likely do not want to do, since it is likely that some people are
#     more likely to receive ORS at any given time step than others
def ors_exposure_effect(exposure, susceptibility_column):
    """Factory that makes function which can be used as the exposure_effect
    for binary categorical risks

    Parameters
    ----------
    exposure : ceam.framework.lookup.TableView
        A lookup for exposure data
    susceptibility_column : str
        The name of the column which contains susceptibility data
    """
    @uses_columns([susceptibility_column, 'diarrhea', 'ors_unit_cost',
                   'ors_cost_to_administer', 'ors_count',
                   'diarrhea_event_count', 'ors_clock'])
    def inner(rates, rr, population_view):
        """
        Parameters
        ----------
        rates:

        rr:

        population_view:

        """
        population = population_view.get(rr.index)

        pop = population.query("diarrhea != 'healthy'").copy()

        exp = exposure(pop.index)

        exp, categories = naturally_sort_df(exp)

        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)

        exp = pop.join(exp)

        exp = assign_exposure_categories(exp, susceptibility_column,
                                         categories)

        df = exp.join(rr)

        df = assign_relative_risk_value(df, categories)

        # costs and counts
        received_ors_index = df.query("exposure_category == 'cat1'").index

        # TODO: Need to bring in the GBD estimates of ORS effectiveness
        df.loc[received_ors_index, 'relative_risk_value'] = 1 - \
            config.ORS.ors_effectiveness

        # TODO: Make sure the categories make sense. Exposure to ORS should
        #     decrease risk (i.e. RR should be less than 1)
        # TODO: Confirm that excess mortality rates are being fed in here
        rates.loc[pop.index] *= (df.relative_risk_value.values)

        # FIXME: ORS clock isn't working properly. This function needs to
        #     happen later in the priority!
        # using this ors_clock variable to make sure ors count and ors costs
        #     are only counted once per bout
        if not pop.loc[received_ors_index].empty:

            received_ors_pop = pop.loc[received_ors_index]
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_unit_cost'] += config.ORS.ORS_unit_cost
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_cost_to_administer'] += config.ORS.cost_to_administer_ORS
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_count'] += 1
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_clock'] = received_ors_pop['diarrhea_event_count']

            population_view.update(received_ors_pop)

        return rates

    return inner


class ORSRiskEffect:
    """RiskEffect objects bundle all the effects that a given risk has on a
    cause.
    """
    def __init__(self, rr_data, exposure_effect):
        """
        Parameters
        ----------
        rr_data : pandas.DataFrame
            A dataframe of relative risk data with age, sex, year, and rr
            columns
        paf_data : pandas.DataFrame
            A dataframe of population attributable fraction data with age, sex,
            year, and paf columns
        cause : str
            The name of the cause to effect as used in named variables like
            'incidence_rate.<cause>'
        exposure_effect : callable
            A function which takes a series of incidence rates and a series of
            relative risks and returns rates modified as appropriate for this
            risk
        """
        self.rr_data = rr_data
        self.exposure_effect = exposure_effect

    def setup(self, builder):
        self.rr_lookup = builder.lookup(self.rr_data)
        builder.modifies_value(self.excess_mortality_rates, 'excess_mortality.diarrhea')

        return [self.exposure_effect]

    # TODO: This was defined as incidence rates, but it just grabs the relative
    #    risk, correct? How do the two return statements factor in?
    # TODO: Incorporate PAFs below
    def excess_mortality_rates(self, index, rates):
        rr = self.rr_lookup(index)

        return self.exposure_effect(rates, rr)


def make_gbd_risk_effects(risk_id, causes, rr_type, effect_function):
    return [ORSRiskEffect(
        get_relative_risks(risk_id=risk_id, cause_id=cause_id,
                           rr_type=rr_type),
                        effect_function)
                        for cause_id, cause_name in causes]


class ORS():
    """
    Determines the change in ORS exposure due to the intervention (change is
    specified in the config file)
    """

    def __init__(self):
        self.active = config.ORS.run_intervention

    def setup(self, builder):

        ors_exposure = get_ors_exposure()

        if self.active:
            # add exposure above baseline increase in intervention scenario
            ors_exposure_increase_above_baseline = config.ORS.ors_exposure_increase_above_baseline
            ors_exposure['cat1'] += ors_exposure_increase_above_baseline
            ors_exposure['cat2'] -= ors_exposure_increase_above_baseline

        self.exposure = builder.value('exposure.ors')

        self.exposure.source = builder.lookup(ors_exposure)

        self.randomness = builder.randomness('ors')

        # FIXME: Update to use the ORS rei id
        effect_function = ors_exposure_effect(self.exposure,
                                              'ors_susceptibility')
        risk_effects = make_gbd_risk_effects(238, [
            (302, 'diarrhea'),
            ], 'mortality', effect_function)

        return risk_effects

    # FIXME: May want to rethink susceptibility column getting assigned at
    #     birth. Distribution of susceptibility may differ for people that
    #     actually get diarrhea
    @listens_for('initialize_simulants')
    @uses_columns(['ors_susceptibility', 'ors_unit_cost',
                   'ors_cost_to_administer', 'ors_count', 'ors_clock'])
    def load_columns(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index),
                                               name='ors_susceptibility',
                                               index=event.index))
        event.population_view.update(pd.DataFrame({'ors_unit_cost': np.zeros(len(event.index),
                                                   dtype=float)},
                                                   index=event.index))
        event.population_view.update(pd.DataFrame({'ors_cost_to_administer': np.zeros(len(event.index),
                                                   dtype=float)},
                                                   index=event.index))
        event.population_view.update(pd.DataFrame({'ors_count': np.zeros(len(event.index),
                                                   dtype=int)},
                                                   index=event.index))
        event.population_view.update(pd.DataFrame({'ors_clock': np.zeros(len(event.index),
                                                   dtype=int)},
                                                   index=event.index))

    # FIXME: Pick the correct priority here. Check diarrhea disease model to make sure this is correct
    @listens_for('time_step', priority=)
    @uses_columns(['ors_propensity', 'diarrhea', 'diarrhea_event_time'], 'alive')
    def set_ors_working_column(self, event):
        pop = event.population

        # if current time > ORS_end_time, set ORS working col to 0


        # filter to people that got diarrhea in the current time step 
        # FIXME: pretty unrealistic if we're assuming that everyone that gets ORS gets it on the first day that they get diarrhea, but also seems necessary to ensure correct exposure
        pop = pop.query("diarrhea_event_time == {}".format(event.time))

        # set an ORS start time and an ORS end time

        exp = self.exposure(pop.index)

        exp, categories = naturally_sort_df(exp)

        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)

        exp = pop.join(exp)

        exp = assign_exposure_categories(exp, self.propensity_column,
                                         categories)

        df = exp.join(self.rr)

        df = assign_relative_risk_value(df, categories)

        # costs and counts
        received_ors_index = df.query("exposure_category == 'cat1'").index

        # TODO: Need to bring in the GBD estimates of ORS effectiveness
        df.loc[received_ors_index, 'relative_risk_value'] = 1 - \
            config.ORS.ors_effectiveness

        # TODO: Make sure the categories make sense. Exposure to ORS should
        #     decrease risk (i.e. RR should be less than 1)
        # TODO: Confirm that excess mortality rates are being fed in here
        rates.loc[pop.index] *= (df.relative_risk_value.values)

        # FIXME: ORS clock isn't working properly. This function needs to
        #     happen later in the priority!
        # using this ors_clock variable to make sure ors count and ors costs
        #     are only counted once per bout
        if not pop.loc[received_ors_index].empty:

            received_ors_pop = pop.loc[received_ors_index]
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_unit_cost'] += config.ORS.ORS_unit_cost
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_cost_to_administer'] += config.ORS.cost_to_administer_ORS
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_count'] += 1
            received_ors_pop.loc[received_ors_pop.ors_clock < received_ors_pop.diarrhea_event_count,
                'ors_clock'] = received_ors_pop['diarrhea_event_count']

            population_view.update(received_ors_pop)

    @modifies_value('metrics')
    @uses_columns(['ors_count', 'ors_unit_cost', 'ors_cost_to_administer'])
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

        metrics['ors_unit_cost'] = population['ors_unit_cost'].sum()
        metrics['ors_count'] = population['ors_count'].sum()
        metrics['ors_cost_to_administer'] = population['ors_cost_to_administer'].sum()

        return metrics


# End.
