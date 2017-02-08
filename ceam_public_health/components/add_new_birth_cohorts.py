from datetime import timedelta

import pandas as pd
import numpy as np

from scipy.stats import norm

from ceam import config

from ceam.framework.state_machine2 import Transition, TransitionSet, record_event_time, active_after_delay, new_state_side_effect
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns, creates_simulants


class AddNewBirthCohort:
    @creates_simulants
    @listens_for('time_step', priority=9)
    def _add_new_birth_cohort(self, event, creator):
        _current_time = pd.Timestamp(event.time)

        _current_time = _current_time.to_datetime()

        _day = _current_time.day

        _month = _current_time.month

        day_month = (_day, _month)

        if  day_month != (1, 6):
            _new_year = 'not new year'
        else:
            _new_year = 'new year'

        if config.get('simulation_parameters', 'number_of_new_simulants_each_year') != '':
            if _new_year == 'new year':
                creator(config.getint('simulation_parameters', 'number_of_new_simulants_each_year'), population_configuration={'initial_age': 0.0, 'year_start': config.getint('simulation_parameters', 'year_start')})


# End.
