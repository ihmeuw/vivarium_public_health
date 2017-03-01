import pytest
import pandas as pd, numpy as np
from ceam import config
from ceam_public_health.components.diarrhea_disease_model import DiarrheaEtiologyState, ApplyDiarrheaExcessMortality
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam.framework.state_machine import State, Transition
from ceam_inputs import get_incidence


def test_DiarrheaEtiologyState():
    model = DiseaseModel('diarrhea')
    healthy = State('healthy', key='diarrhea')

    etiology_state = DiarrheaEtiologyState('diarrhea', key='diarrhea', disability_weight=.99)

    transition = RateTransition(etiology_state,
                                'diarrhea',
                                build_table(0))

    healthy.transition_set.append(transition)

    model.states.extend([healthy, etiology_state])

    simulation = setup_simulation([generate_test_population, model])

    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 1

    transition.base_incidence = simulation.tables.build_table(inc)

    incidence_rate = simulation.values.get_rate('incidence_rate.diarrhea')

    pump_simulation(simulation, iterations=1)

    dis_weight = simulation.values.get_value('disability_weight')

    only_men = simulation.population.population.query("sex == 'Male'")
    only_women = simulation.population.population.query("sex == 'Female'")

    assert pd.unique(dis_weight(only_men.index)) == [.99], "DiarrheaEtiologyState needs to assign the disability weight that is provided to people that get diarrhea"
    assert pd.unique(dis_weight(only_women.index)) == [0], "DiarrheaEtiologyState needs to assign the disability weight that is provided to people that get diarrhea"


def test_ApplyDiarrheaExcessMortality():
    model = DiseaseModel('diarrhea')
    healthy = State('healthy', key='diarrhea')

    mortality_state = ApplyDiarrheaExcessMortality(excess_mortality_data=build_table(0.7), cause_specific_mortality_data=build_table(0.0))

    healthy.transition_set.append(Transition(mortality_state))

    model.states.extend([healthy, mortality_state])

    simulation = setup_simulation([generate_base_population, model])

    mortality_rate = simulation.values.get_rate('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0))

    pump_simulation(simulation, iterations=1)

    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index))


# End.
