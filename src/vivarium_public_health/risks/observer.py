import pandas as pd


class CategoricalRiskObserver:
    configuration_defaults = {
        'observer': {
            'risk': {
                'type': 'risk_factor',
                'name': 'risk',
                'by_year': True
            }
        }
    }

    def __init__(self):
        self.configuration_defaults = CategoricalRiskObserver.configuration_defaults

    def setup(self, builder):
        self.risk = self.configuration_defaults.observer.risk.name
        self.clock = builder.time.clock()
        self.start_time = self.clock()
        self.population_view = builder.population.get_view(['alive', 'age'])
        self.age_bins = (builder.data.load('population.structure')[['age_group_start', 'age_group_end']]
                         .drop_duplicates()
                         .reset_index(drop=True))
        if builder.configuration.population.exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_end <= builder.configuration.population.exit_age]

        self.exposure = builder.value.get_vlaue(f'{self.risk}.exposure')
        categories = builder.data.load(f'{self.configuration_defaults.observer.risk.type}.{self.risk}.categories')
        frame_dict = {f'{self.risk}_{cat}': 0 for cat in categories}
        self.proportion = pd.DataFrame(frame_dict, index=self.age_bins.index)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        current_year = event.time.year
        midpoint = pd.datetime(current_year, 7, 1)
        if event.time < midpoint <= event.time + event.step_size:
            import pdb; pdb.set_trace()
            exposure_proportion = self.exposure(pop.index)







