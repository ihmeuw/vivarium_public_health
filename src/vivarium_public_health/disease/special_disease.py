import pandas as pd


class RiskAttributableDisease:
    """
    configuration_defaults for categorical risk should be the list of categories to be in this state
    """

    configuration_defaults = {
        'risk_attributable_disease': {
            'threshold': None,
            'mortality': True
        }
    }

    def __init__(self, name, risk):
        self.name = name
        self.risk = risk
        self.configuration_defaults = {
            self.name: RiskAttributableDisease.configuration_defaults['risk_attributable_disease']
        }

    def setup(self, builder):
        self.threshold = builder.configuration[self.name].threshold
        disability_weight = builder.data.load(f'cause.{self.name}.disability_weight')
        self.distribution = builder.data.load(f'risk_factor.{self.risk}.distribution')

        if builder.configuration[self.name].mortality:
            csmr_data = builder.data.load(f'cause.{self.name}.cause_specific_mortality')
            builder.value.register_value_modifier('csmr_data', lambda: csmr_data)
            excess_mortality_data = builder.data.load(f'cause.{self.name}.excess_mortality')
            builder.value.register_value_modifier('mortality_rate', self.mortality_rates)
            self._mortality = builder.value.register_value_producer(
                f'{self.name}.excess_mortality', source=builder.lookup.build_table(excess_mortality_data)
            )

        self._disability_weight = builder.lookup.build_table(disability_weight)
        self.disability_weight = builder.value.register_value_producer(f'{self.name}.disability_weight',
                                                                       source=self.compute_disability_weight)
        builder.value.register_value_modifier('disability_weight', modifier=self.disability_weight)
        self.exposure = builder.value.get_value(f'{self.risk}.exposure')

        self.population_view = builder.population.get_view([self.name])
        builder.population.initializes_simulants(self.initialize_simulants)
        builder.event.register_listener('time_step', self.on_time_step)

    def filter_by_exposure(self, index):
        import pdb; pdb.set_trace()
        exposure = self.exposure(index)
        if self.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
            sick = exposure[exposure.isin(self.threshold)]
        else:
            sick = exposure > self.threshold
        return sick

    def initialize_simulants(self, pop_data):
        import pdb; pdb.set_trace()
        new_pop = pd.Series(f'susceptible_to_{self.name}', index=pop_data.index, name=self.name)
        sick = self.filter_by_exposure(pop_data.index)
        new_pop[sick] = self.name
        self.population_view.update(new_pop)

    def on_time_step(self, event):
        import pdb;
        pdb.set_trace()
        pop = self.population_view.get(event.index, query='alive == "alive"')
        sick = self.filter_by_exposure(event.index)
        pop.loc[sick, self.name] = self.name
        self.population_view.update(pop)

    def mortality_rates(self, index, rates_df):
        import pdb;
        pdb.set_trace()
        population = self.population_view.get(index)
        rate = (self._mortality(population.index, skip_post_processor=True) * (population[self.name] == self.name))
        rates_df[self.name] = rate
        return rates_df

    def compute_disability_weight(self, index):
        import pdb;
        pdb.set_trace()
        population = self.population_view.get(index)
        return self._disability_weight(population.index) * ((population[self.name] == self.name) &
                                                            population.alive == 'alive')
