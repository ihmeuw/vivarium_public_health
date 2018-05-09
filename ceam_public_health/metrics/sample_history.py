import pandas as pd


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
        self.config = builder.configuration.sample_history
        self.run_id = builder.configuration.run_configuration.run_id
        self.randomness = builder.randomness.get_stream('sample_history')
        self.population_view = builder.population.get_view(columns=[])  # [] actually means "all"
        builder.population.initializes_simulants(self.load_population_columns)
        self.key = '/{}-{}'.format(builder.configuration.opportunistic_screening.medication_sheet, self.run_id)
        if self.key == '/':
            self.key += 'base'

        builder.event.register_listener('collect_metrics', self.record)
        builder.event.register_listener('simulation_end', self.dump)

    def load_population_columns(self, pop_data):
        sample_size = self.config.sample_size
        if sample_size is None or sample_size > len(pop_data.index):
            sample_size = len(pop_data.index)
        draw = self.randomness.get_draw(pop_data.index)
        priority_index = [i for d, i in sorted(zip(draw, pop_data.index), key=lambda x:x[0])]
        self.sample_index = priority_index[:sample_size]

    def record(self, event):
        sample = self.population_view.get(event.index).loc[self.sample_index]

        self.sample_frames[event.time] = sample

    def dump(self, event):
        # NOTE: I'm suppressing two very noisy warnings about HDF writing that I don't think are relevant to us
        import warnings
        import tables
        from pandas.core.common import PerformanceWarning
        warnings.filterwarnings('ignore', category=PerformanceWarning)
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        pd.Panel(self.sample_frames).to_hdf(self.config.path, key=self.key)
