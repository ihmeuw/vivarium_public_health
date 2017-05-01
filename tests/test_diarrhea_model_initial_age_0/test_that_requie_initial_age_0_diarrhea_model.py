import pytest
import pandas as pd, numpy as np
from ceam import config
from ceam_public_health.components.diarrhea_disease_model import DiarrheaEtiologyState, DiarrheaBurden, diarrhea_factory
from ceam_tests.util import build_table, setup_simulation, generate_test_population, pump_simulation
from ceam_public_health.components.disease import DiseaseModel, RateTransition
from ceam.framework.state_machine import State, Transition
from ceam_inputs import get_etiology_specific_incidence, get_excess_mortality, get_cause_specific_mortality, get_duration_in_days, get_cause_deleted_mortality_rate, get_cause_specific_mortality
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam_public_health.components.accrue_susceptible_person_time import AccrueSusceptiblePersonTime
from ceam.framework.event import Event
from datetime import timedelta, datetime
from ceam_public_health.components.base_population import Mortality
from ceam_inputs.gbd_ms_functions import get_disability_weight


def test_cause_deletion():
    factory = diarrhea_factory()
    simulation = setup_simulation([generate_test_population, Mortality()] + factory, start=datetime(2005, 1, 1))

    # determine what the cause-deleted mortality rate should be
    cause_deleted_mr = get_cause_deleted_mortality_rate([get_cause_specific_mortality(1181)])

    # get the mortality rate from the simulation
    simulation_mortality_rate = simulation.values.get_rate('mortality_rate')

    # compare for the earliest age group (this test requires that generate_test_population is set to create a cohort of newborns)
    cause_deleted_mr_values = cause_deleted_mr.query("year==2005 and age<.01").cause_deleted_mortality_rate.values
    simulation_values = simulation_mortality_rate(simulation.population.population.index).death_due_to_other_causes.unique()

    # check that the value in the simulation is what it should be
    assert np.allclose((cause_deleted_mr_values * 1/365), (simulation_values), atol=.01), "ensure that the mortality rate at the start of the simulation is the diarrhea cause-deleted mortality rate"

