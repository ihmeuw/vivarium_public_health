import os

import pandas as pd

from vivarium.framework.values import produces_value
from vivarium.framework.event import listens_for, emits
from vivarium.framework.util import collapse_nested_dict

from ceam_inputs import get_age_bins

import logging

_log = logging.getLogger(__name__)


class EpidemiologicalMeasures:
    """ Gathers measures (prevalence, incidence rates, mortality rates, etc)
    from other components in the system and saves them to an HDF file which
    can be further analyzed. For example by ceam_public_health/scripts/measure_analysis.py
    """
    def setup(self, builder):
        self.run_config = builder.configuration.run_configuration
        self.point_measures = builder.value('epidemiological_point_measures')
        self.span_measures = builder.value('epidemiological_span_measures')
        self.clock = builder.clock()

        self.run_key = self.run_config.run_key.to_dict() if 'run_key' in self.run_config else None

        results_directory = self.run_config.results_directory if 'results_directory' in self.run_config else '/tmp'
        results_directory = os.path.join(results_directory, 'epidemiological_measures')
        os.makedirs(results_directory, exist_ok=True)
        self.output_path = os.path.join(results_directory, 'measure_{}.hdf'.format(self.run_config.run_id))

        self.collecting = False
        self.last_collected_year = -1

    @produces_value('epidemiological_point_measures')
    @produces_value('epidemiological_span_measures')
    def base_cube(self, index, age_groups, sexes, all_locations, duration):
        return pd.DataFrame(
            columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value', 'sample_size']
        ).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause'])

    @listens_for('collect_metrics')
    @emits('begin_epidemiological_measure_collection')
    def time_step(self, event, event_emitter):
        mid_year = pd.Timestamp(year=event.time.year, month=7, day=2)
        year_start = pd.Timestamp(year=event.time.year, month=1, day=1)

        if self.collecting:
            year_end = pd.Timestamp(year=self.last_collected_year, month=12, day=31)
            # On the year following a GBD year, reel the data in
            if self.clock() < year_end <= event.time:
                _log.debug('end collection')
                self.dump_measures(event.index, self.last_collected_year)
                self.collecting = False

        if event.time.year % 5 == 0:  # FIXME: If year in list of GBD years
            if self.clock() < mid_year <= event.time:
                # Collect point measures at the midpoint of every gbd year
                self.dump_measures(event.index, event.time.year, point=True)

            if (self.clock() < year_start <= event.time
                    and event.time.year > self.last_collected_year
                    and not self.collecting):
                # Emit the begin collection event every gbd year
                event_emitter(event.split(event.index))
                _log.debug('begin collection')
                self.collecting = True
                self.last_collected_year = event.time.year

    @listens_for('post_setup')
    def prepare_output_file(self, event):
        pd.DataFrame().to_hdf(self.output_path, 'data')

    def dump_measures(self, index, current_year, point=False):
        age_group_ids = list(range(2, 22))
        age_groups = get_age_bins().query('age_group_id in @age_group_ids')
        age_groups = age_groups[['age_group_years_start', 'age_group_years_end']].values
        if point:
            measures = self.point_measures
            _log.debug('collecting point measures')
        else:
            measures = self.span_measures
        df = measures(index, age_groups, ['Male', 'Female'], False, pd.Timedelta(days=365)).reset_index()
        df['year'] = current_year
        df['input_draw'] = self.run_config.input_draw_number
        df['model_draw'] = self.run_config.model_draw_number
        existing_df = pd.read_hdf(self.output_path)
        df = existing_df.append(df)

        if self.run_key:
            for k, v in collapse_nested_dict(self.run_key):
                df[k] = v

        # Enforce some types that occasionally get mis-set
        df['sample_size'] = df.sample_size.astype(int)
        df['sex'] = df.sex.astype(str)
        df['age_low'] = df.age_low.astype(float)
        df['age_high'] = df.age_high.astype(float)

        df.to_hdf(self.output_path, 'data')
