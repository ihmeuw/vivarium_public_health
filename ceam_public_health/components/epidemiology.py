from datetime import timedelta, datetime

import pandas as pd

from ceam.framework.values import produces_value
from ceam.framework.event import listens_for, emits

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
        self.point_measures = builder.value('epidemiological_point_measures')
        self.span_measures = builder.value('epidemiological_span_measures')

        if 'epidemiology' not in config:
            config['epidemiology'] = {}
        self.output_path = config['epidemiology'].get('path', '/tmp/measures.hdf')
        self.collecting = False
        self.last_collected_year = -1

    @produces_value('epidemiological_point_measures')
    @produces_value('epidemiological_span_measures')
    def base_cube(self, index, age_groups, sexes, all_locations, duration):
        return pd.DataFrame(columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value', 'sample_size']).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause'])

    @listens_for('time_step')
    @emits('begin_epidemiological_measure_collection')
    def time_step(self, event, event_emitter):
        time_step = timedelta(days=config.simulation_parameters.time_step)
        mid_year = datetime(year=event.time.year, month=6, day=1)
        year_start = datetime(year=event.time.year, month=1, day=1)


        if self.collecting:
            # On the year following a GBD year, reel the data in
            year_end = datetime(year=self.last_collected_year, month=12, day=31)
            if year_end > event.time - time_step and year_end <= event.time and self.collecting:
                _log.debug('end collection')
                self.dump_measures(event.index)
                self.collecting = False

        if event.time.year % 5 == 0:
            if mid_year > event.time - time_step and mid_year <= event.time:
                # Collect point measures at the midpoint of every gbd year
                self.dump_measures(event.index, point=True)

            if year_start > event.time - time_step and year_start <= event.time and \
               event.time.year > self.last_collected_year and not self.collecting:
                # Emit the begin collection event every gbd year
                event_emitter(event.split(event.index))
                _log.debug('begin collection')
                self.collecting = True
                self.last_collected_year = event.time.year

    @listens_for('post_setup')
    def prepare_output_file(self, event):
        pd.DataFrame().to_hdf(self.output_path, 'data')

    def dump_measures(self, index, point=False):
        age_group_ids = list(range(2,22))
        age_groups = get_age_bins().query('age_group_id in @age_group_ids')
        age_groups = age_groups[['age_group_years_start', 'age_group_years_end']].values
        if point:
            measures = self.point_measures
            _log.debug('collecting point measures')
        else:
            measures = self.span_measures
        df = measures(index, age_groups, ['Male', 'Female'], False, timedelta(days=365)).reset_index()
        df['year'] = self.last_collected_year
        df['draw'] = config.run_configuration.draw_number
        existing_df = pd.read_hdf(self.output_path)
        df = existing_df.append(df)
        df.to_hdf(self.output_path, 'data')
