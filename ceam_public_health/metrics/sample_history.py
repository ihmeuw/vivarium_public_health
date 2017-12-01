import pandas as pd

from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns


class SampleHistory:
    """Collect a detailed record of events that happen to a sampled sub-population

    For use with visualization or analysis. The records are written to an HDF file.
    """

    configuration_defaults = {
            'sample_history': {
                'sample_size': 10000,
                'path': '/tmp/sample.hdf',
            }
    }

    def __init__(self):
        self.sample_frames = {}
        self.sample_index = []

    def setup(self, builder):
        import random, time
        time.sleep(random.random()*60*10)
        self.config = builder.configuration.sample_history
        self.run_id = builder.configuration.run_configuration.run_id
        self.randomness = builder.randomness('sample_history')

        self.key = '/{}-{}'.format(builder.configuration.opportunistic_screening.medication_sheet, self.run_id)
        if self.key == '/':
            self.key += 'base'

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        sample_size = self.config.sample_size
        if sample_size is None or sample_size > len(event.index):
            sample_size = len(event.index)
        draw = self.randomness.get_draw(event.index)
        priority_index = [i for d,i in sorted(zip(draw,event.index), key=lambda x:x[0])]
        self.sample_index = priority_index[:sample_size]

    @listens_for('collect_metrics')
    @uses_columns(None)
    def record(self, event):
        sample = event.population.loc[self.sample_index]

        self.sample_frames[event.time] = sample

    @listens_for('simulation_end')
    def dump(self, event):
        # NOTE: I'm suppressing two very noisy warnings about HDF writing that I don't think are relevant to us
        import warnings
        import tables
        from pandas.core.common import PerformanceWarning
        warnings.filterwarnings('ignore', category=PerformanceWarning)
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        pd.Panel(self.sample_frames).to_hdf(self.config.path, key=self.key)
