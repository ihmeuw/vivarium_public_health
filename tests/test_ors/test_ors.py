import pandas as pd
import numpy as np
import pytest
from datetime import timedelta, datetime

from ceam import config
from ceam.framework.event import Event
from ceam_tests.util import (build_table, setup_simulation,
                                     generate_test_population, pump_simulation)

from ceam_inputs import get_disability_weight
from ceam_inputs import (get_severity_splits, get_cause_specific_mortality,
                                 get_cause_deleted_mortality_rate)

from ceam_public_health.components.base_population import Mortality
from ceam_public_health.components.diarrhea_disease_model import (DiarrheaEtiologyState,
                                                                          DiarrheaBurden,
                                                                          diarrhea_factory)

from ceam_public_health.components.interventions.ors_supplementation_rewrite import ORS


def setup():
    # Remove user overrides but keep custom cache locations if any
    try:
        config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                      'input_data.auxiliary_data_folder'])
    except KeyError:
        pass

    config.simulation_parameters.set_with_metadata('year_start', 2010, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('year_end', 2015, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('time_step', 1, layer='override', source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('initial_age', 0, layer='override',
                                                   source=os.path.realpath(__file__))
    config.simulation_parameters.set_with_metadata('location_id', 179, layer='override',
                                                   source=os.path.realpath(__file__))
    config.ORS.set_with_metadata('ors_exposure_increase_above_baseline', .25, layer='override', source=os.path.realpath(__file__))


def test_determine_who_gets_ors():
    """
    Ensure that the correct number of simulants receive ORS
    """
    factory = diarrhea_factory()

    ors_instance = ORS()

    population_size = 10000

    simulation = setup_simulation([generate_test_population, ors_instance] + factory, population_size=population_size)

    # make it so that all men will get incidence due to rotaviral entiritis
    inc = build_table(0)

    inc.loc[inc.sex == 'Male', 'rate'] = 14000

    rota_inc = simulation.values.get_rate('incidence_rate.diarrhea_due_to_rotaviral_entiritis')

    rota_inc.source = simulation.tables.build_table(inc)

    pump_simulation(simulation, iterations=1)

    pop = simulation.population.population

    ors_proportion = len(pop.query("ors_working == 1 and sex == 'Male'"))/population_size

    no_ors_proportion = len(pop.query("ors_working == 0 sex == 'Male'"))/population_size

    ors_exposure = get_ors_exposures()

    proportion_exposed = ors_exposure.query("sex == 'Male' and age <.01").set_index(['year']).get_value(2010, 'cat2')
    proportion_not_exposed = ors_exposure.query("sex == 'Male' and age <.01").set_index(['year']).get_value(2010, 'cat1')

    assert np.allclose(ors_proportion, proportion_exposed, rtol=.05), "proportion of people on ors should accurately reflect the exposure from GBD"
    assert np.allclose(no_ors_proportion, proportion_not_exposed, rtol=.05), "proportion of people on ors should accurately reflect the exposure from GBD"


def test_ors_working_column():
    """
    Test that the ors working column is only set for people that should receive ORS and only in the correct time window
    """


def test_mortality_rates():
    """
    Test that people who are unexposed to ORS and have severe diarrhea have the severe diarrhea mortality rate * (1 - PAF) and people that have severe diarrhea and get ORS have the severe diarrhea mortality rate * (1 - PAF) * rr
    """

