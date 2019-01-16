import pandas as pd
from vivarium_public_health.risks.data_transformation import split_risk_from_type

class CategoricalRiskObserver:
    """ An observer for a categorical risk factor.
    This component by default observes proportion of simulants in each age
    group who are alive and in each category of risk at the midpoint of a year
    unless the sample date is specified in the configuration.

    It also collects the total number of alive simulants in each age group
    when the proportion is collected.

    """
    configuration_defaults = {
        'observer': {
            'risk': {
                'by_year': True,
                'sample_date': {
                    'month': 7,
                    'day': 1
                }
            }
        }
    }

    def __init__(self, risk):
        """
        Parameters
        ----------
        risk :
        the type and name of a risk, specified as "type.name". Type is singular.

        """
        self.risk_type, self.risk_name = split_risk_from_type(risk)
        self.configuration_defaults = CategoricalRiskObserver.configuration_defaults

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.population_view = builder.population.get_view(['alive', 'age'])
        self.age_bins = builder.data.load('population.age_bins')

        if builder.configuration.population.exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_end <= builder.configuration.population.exit_age]

        self.sample_date_config = builder.configuration.observer.risk.sample_date
        self.exposure = builder.value.get_value(f'{self.risk_name}.exposure')
        self.categories = builder.data.load(f'{self.risk_type}.{self.risk_name}.categories')
        self.data = pd.DataFrame()

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        pop = pop[pop.alive == 'alive']
        current_year = event.time.year
        sample_date = pd.datetime(current_year, self.sample_date_config.month, self.sample_date_config.day)

        if self.clock() <= sample_date < event.time:
            frame_dict = {f'{self.risk_name}_{cat}': 0 for cat in self.categories}
            exposure_proportion = pd.DataFrame(frame_dict, index=self.age_bins.index)
            exposure_proportion['year'] = current_year
            exposure_proportion['alive_simulants_in_age_group'] = 0
            for group, age_group in self.age_bins.iterrows():
                start, end = age_group.age_group_start, age_group.age_group_end
                in_group = pop[(pop.age >= start) & (pop.age < end)]
                exposure_in_group = self.exposure(in_group.index).value_counts()
                exposure_proportion.loc[group, 'alive_simulants_in_age_group'] = len(in_group)
                for cat, count in exposure_in_group.iteritems():
                    exposure_proportion.loc[group, f'{self.risk_name}_{cat}'] = count/len(in_group)
            self.data = self.data.append(exposure_proportion)

    def metrics(self, index, metrics):
        result = self.age_bins.join(self.data).drop(['age_group_start', 'age_group_end'], axis=1)
        result = pd.melt(result, id_vars=['age_group_name', 'year'])
        for _, row in result.iterrows():
            metrics[f'age_group_{row.age_group_name.replace(" ", "_")}_year_{row.year}_{row.variable}'] = row.value

        return metrics
