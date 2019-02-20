import pandas as pd


class RiskAttributableDisease:
    """ Disease fully attributed by a risk.
    This is for some (risk, cause) pairs with population attributable fraction
    equal to 1 where `infected to the cause` is defined by the level of risk
    exposure higher than the threshold level.

    For example, one who has Fasting plasma glucose of greater than 7 mmol/L
    is considered to have `diabetes_mellitus`. Another example is
    `protein_energy_malnutrition`. One who is exposed to child wasting of cat1
    or cat2 become infected to `protein_energy_malnutrition`.

    Configuration defaluts should be given as, for the continuous risk factor,

    diabetes_mellitus:
        threshold : 7
        mortality : True
        absorbing_state : True # once get infected, cannot be recovered.

    For the categorical risk factor,

    protein_energy_malnutrition:
        threshold : ['cat1', 'cat2'] # provide the categories to get PEM.
        mortality : True
        absorbing_state : False # can be recovered.

    """

    configuration_defaults = {
        'risk_attributable_disease': {
            'threshold': None,
            'mortality': True,
            'absorbing_state': False
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
        self.absorbing_state = builder.configuration[self.name].absorbing_state
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
        self.population_view = builder.population.get_view([self.name, 'alive'])

        builder.event.register_listener('time_step', self.on_time_step)
        builder.population.initializes_simulants(self.on_initialize_simulants)

    def filter_by_exposure(self, index):

        exposure = self.exposure(index)
        if self.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
            sick = exposure.isin(self.threshold)
        else:
            sick = exposure > self.threshold
        return sick

    def on_initialize_simulants(self, pop_data):
        new_pop = pd.Series(f'susceptible_to_{self.name}', index=pop_data.index, name=self.name)
        sick = self.filter_by_exposure(pop_data.index)
        new_pop[sick] = self.name
        self.population_view.update(new_pop)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        sick = self.filter_by_exposure(event.index)
        if not self.absorbing_state:
            pop.loc[~sick, self.name] = f'susceptible_to_{self.name}'
        pop.loc[sick, self.name] = self.name
        self.population_view.update(pop)

    def mortality_rates(self, index, rates_df):
        population = self.population_view.get(index)
        rate = (self._mortality(population.index, skip_post_processor=True)
                * (population[self.name] == self.name))
        if isinstance(rates_df, pd.Series):
            rates_df = pd.DataFrame({rates_df.name: rates_df, self.name: rate})
        else:
            rates_df[self.name] = rate
        return rates_df

    def compute_disability_weight(self, index):
        population = self.population_view.get(index, query=f'alive=="alive" and {self.name}=="{self.name}"')
        return self._disability_weight(population.index)
