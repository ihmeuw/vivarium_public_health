

class TreatmentEffect:

    def __init__(self, treatment, cause):
        self.treatment = treatment
        self.affected_cause = cause

    def setup(self, builder):
        builder.value.register_value_modifier(f'{self.affected_cause}.incidence_rate',
                                              modifier=self.incidence_rates)
        self.population_view = builder.population.get_view(['alive'])

    def incidence_rates(self, index, rates):
        """Modifies the incidence of shigellosis.

        Parameters
        ----------
        index: pandas.Index
            The set of simulants who are susceptible to shigellosis.
        rates: pandas.Series
            The baseline incidence rate of shigellosis.

        Returns
        -------
        pandas.Series
            The shigellosis incidence rates adjusted for the presence of the vaccine.
        """
        population = self.population_view.get(index)
        population = population[population.alive == 'alive']
        protection = self.treatment.determine_protection(population)
        rates.loc[population.index] *= (1-protection.values)
        return rates
