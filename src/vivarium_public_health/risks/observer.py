import pandas as pd
from vivarium_public_health.risks.data_transformations import RiskString


class CategoricalRiskObserver:
    """ An observer for a categorical risk factor.
    This component by default observes proportion of simulants in each age
    group who are alive and in each category of risk at the midpoint of a year
    unless the sample date is specified in the configuration.

    It also collects the total number of alive simulants in each age group
    when the proportion is collected.

    configuration should be given as e.g.,

    {risk_name}_observer:
        sample_date:
            'month' : 12
            'day': 31
    """
    configuration_defaults = {
        'risk_observer': {
            'sample_date': {
                'month': 7,
                'day': 1
            }
        }
    }

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
        the type and name of a risk, specified as "type.name". Type is singular.

        """
        self.risk = RiskString(risk)
        self.configuration_defaults = {
            f'{self.risk.name}_observer': CategoricalRiskObserver.configuration_defaults['risk_observer']
        }

    def setup(self, builder):
        self.config = builder.configuration[f'{self.risk.name}_observer']
        self.data = {}

        self.clock = builder.time.clock()

        self.categories = builder.data.load(f'{self.risk}.categories')
        self.age_bins = builder.data.load('population.age_bins')
        exit_age = builder.configuration.population.exit_age
        if exit_age:
            self.age_bins = self.age_bins[self.age_bins.age_group_start < exit_age]
            self.age_bins.loc[self.age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age

        self.population_view = builder.population.get_view(['alive', 'age'], query='alive == "alive"')

        self.exposure = builder.value.get_value(f'{self.risk.name}.exposure')
        builder.value.register_value_modifier('metrics', self.metrics)

        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)

        if self.should_sample(event.time):
            sample = self.generate_sampling_frame()
            exposure = self.exposure(pop.index)

            for group, age_group in self.age_bins.iterrows():
                start, end = age_group.age_group_start, age_group.age_group_end
                in_group = pop[(pop.age >= start) & (pop.age < end)]
                sample.loc[group] = exposure.loc[in_group.index].value_counts()

            self.data[self.clock().year] = sample

    def should_sample(self, event_time: pd.Timestamp):
        sample_date = pd.Timestamp(event_time.year, self.config.sample_date.month, self.config.sample_date.day)
        return self.clock() <= sample_date < event_time

    def generate_sampling_frame(self):
        sample = pd.DataFrame({f'{cat}': 0 for cat in self.categories}, index=self.age_bins.index)
        return sample

    def metrics(self, index, metrics):
        for age_id, age_group in self.age_bins.iterrows():
            age_group_name = age_group.age_group_name.replace(" ", "_").lower()
            for year, sample in self.data.items():
                for category in sample.columns:
                    label = f'{self.risk.name}_{category}_exposed_in_{year}_among_{age_group_name}'
                    metrics[label] = sample.loc[age_id, category]
        return metrics
