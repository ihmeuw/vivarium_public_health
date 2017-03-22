from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from ceam.framework.values import produces_value
from ceam.framework.event import listens_for, emits
from ceam.framework.population import uses_columns

from ceam import config

from ceam_inputs import get_age_bins

import logging
_log = logging.getLogger(__name__)

class EpidemiologicalMeasures:
    """ Gathers measures (prevalence, incidence rates, mortality rates, etc)
    from other components in the system and saves them to an HDF file which
    can be further analyzed. For example by ceam_public_health/scripts/measure_analysis.py
    """
    def setup(self, builder):
        self.measures = builder.value('epidemiological_measures')

        if 'epidemiology' not in config:
            config['epidemiology'] = {}
        self.output_path = config['epidemiology'].get('path', '/tmp/measures.hdf')
        self.collecting = False

    @produces_value('epidemiological_measures')
    def base_cube(self, index, age_groups, sexes, all_locations, duration):
        return pd.DataFrame(columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value', 'sample_size']).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause'])

    @listens_for('time_step')
    @emits('begin_epidemiological_measure_collection')
    def time_step(self, event, event_emitter):
        time_step = timedelta(days=config.getfloat('simulation_parameters', 'time_step'))
        mid_year = datetime(year=event.time.year, month=6, day=1)

        # Emit the begin collection event every gbd year
        if event.time.year % 5 == 0:
            if mid_year > event.time - time_step and mid_year <= event.time:
                event_emitter(event.split(event.index))
                _log.debug('begin collection')
                self.collecting = True

        # On the year following a GBD year, reel the data in
        if (event.time.year - 1) % 5 == 0 and self.collecting:
            if mid_year > event.time - time_step and mid_year <= event.time:
                self.dump_measures(event)

    @listens_for('post_setup')
    def prepare_output_file(self, event):
        pd.DataFrame().to_hdf(self.output_path, 'data', format='t')

    @listens_for('simulation_end')
    def dump_measures(self, event):
        if self.collecting:
            _log.debug('end collection')
            age_group_ids = list(range(2,22))
            age_groups = get_age_bins().query('age_group_id in @age_group_ids')
            age_groups = age_groups[['age_group_years_start', 'age_group_years_end']].values
            df = self.measures(event.index, age_groups, ['Male', 'Female'], False, timedelta(days=365)).reset_index()
            df['year'] = event.time.year - 1
            df['draw'] = config.getint('run_configuration', 'draw_number')
            df.to_hdf(self.output_path, 'data', format='t', append=True)
            self.collecting = False
