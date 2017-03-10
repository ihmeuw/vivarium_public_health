import xarray as xr
import numpy as np
import pandas as pd

from ceam.framework.values import produces_value
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam import config

from ceam_inputs import get_age_bins

class EpidemiologicalMeasures:
    def setup(self, builder):
        self.measures = builder.value('epidemiological_measures')

        if 'epidemiology' not in config:
            config['epidemiology'] = {}
        self.output_path = config['epidemiology'].get('path', '/tmp/measures.hdf')

    @produces_value('epidemiological_measures')
    def base_cube(self, index, age_groups, sexes, all_locations):
        return pd.DataFrame(columns=['measure', 'age_low', 'age_high', 'sex', 'location', 'cause', 'value']).set_index(['measure', 'age_low', 'age_high', 'sex', 'location', 'cause'])

    @listens_for('simulation_end')
    @uses_columns([], 'alive')
    def dump_measures(self, event):
        age_group_ids = list(range(2,22))
        age_groups = get_age_bins().query('age_group_id in @age_group_ids')
        age_groups = age_groups[['age_group_years_start', 'age_group_years_end']].values
        df = self.measures(event.index, age_groups, ['Male', 'Female'], False).reset_index()
        df['year'] = event.time.year
        df['draw'] = config.getint('run_configuration', 'draw_number')
        df.to_hdf(self.output_path, 'data')
