import pytest
import pandas as pd, numpy as np
from ceam import config
from ceam_public_health.components.diarrhea_disease_model import DiarrheaEtiologyState, ApplyDiarrheaExcessMortality, ApplyDiarrheaRemission
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam.framework.state_machine import State, Transition
from ceam_inputs import get_etiology_specific_incidence, get_excess_mortality, get_cause_specific_mortality, get_duration_in_days
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_public_health.components.accrue_susceptible_person_time import AccrueSusceptiblePersonTime

list_of_etiologies = ['diarrhea_due_to_shigellosis']


def test_diarrhea_model():

    states_dict = {}

    transition_dict = {}

    dict_of_etiologies_and_eti_risks = {'shigellosis': 175}

    for key, value in dict_of_etiologies_and_eti_risks.items():

        diarrhea_due_to_pathogen = 'diarrhea_due_to_shigellosis'

        module = DiseaseModel(diarrhea_due_to_pathogen)

        healthy = State('healthy', key=diarrhea_due_to_pathogen)

        etiology_state = DiarrheaEtiologyState(diarrhea_due_to_pathogen, key=diarrhea_due_to_pathogen, disability_weight=0.99)

        inc = build_table(0)

        inc.loc[inc.sex == 'Male', 'rate'] = 1

        transition = RateTransition(etiology_state,
                                    diarrhea_due_to_pathogen,
                                    inc)

        healthy.transition_set.append(transition)

        module.states.extend([healthy, etiology_state])


    @listens_for('initialize_simulants')
    @uses_columns(['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time', 'diarrhea_event_end_time'])
    def _create_diarrhea_column(event):

        length = len(event.index)

        event.population_view.update(pd.DataFrame({'diarrhea': ['healthy']*length}, index=event.index))
        event.population_view.update(pd.DataFrame({'diarrhea_event_count': np.zeros(len(event.index), dtype=int)}, index=event.index))

        event.population_view.update(pd.DataFrame({'diarrhea_event_time': [pd.NaT]*length}, index=event.index))
        event.population_view.update(pd.DataFrame({'diarrhea_event_end_time': [pd.NaT]*length}, index=event.index))


    @listens_for('time_step', priority=6)
    @uses_columns(['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time'] + list_of_etiologies + [i + '_event_count' for i in list_of_etiologies])
    def _move_people_into_diarrhea_state(event):
        """
        Determines who should move from the healthy state to the diarrhea state and counts both cases of diarrhea and cases of diarrhea due to specific etiologies
        """

        pop = event.population_view.get(event.index)

        pop = pop.query("diarrhea == 'healthy'")

        for etiology in list_of_etiologies:

            pop.loc[pop['{}'.format(etiology)] == etiology, 'diarrhea'] = 'diarrhea'
            pop.loc[pop['{}'.format(etiology)] == etiology, '{}_event_count'.format(etiology)] += 1

        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_count'] += 1

        # set diarrhea event time here
        pop.loc[pop['diarrhea'] == 'diarrhea', 'diarrhea_event_time'] = pd.Timestamp(event.time)

        event.population_view.update(pop[['diarrhea', 'diarrhea_event_count', 'diarrhea_event_time'] + [i + '_event_count' for i in list_of_etiologies]])


    excess_mort = ApplyDiarrheaExcessMortality(get_excess_mortality(1181), get_cause_specific_mortality(1181))

    remission = ApplyDiarrheaRemission(get_duration_in_days(1181))

    list_of_module_and_functs = [module, _move_people_into_diarrhea_state, _create_diarrhea_column, excess_mort, remission, AccrueSusceptiblePersonTime('diarrhea', 'diarrhea')]

    simulation = setup_simulation([generate_test_population, list_of_module_and_functs])

    pump_simulation(simulation, iterations=1) 

    # now that the simulation is set up, write a bunch of tests!
    dis_weight = simulation.values.get_value('disability_weight')

    assert pd.unique(dis_weight(only_men.index)) == [.99], "DiarrheaEtiologyState needs to assign the disability weight that is provided to people that get diarrhea"

# End.
