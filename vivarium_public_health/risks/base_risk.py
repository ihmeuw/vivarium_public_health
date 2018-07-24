import pandas as pd

from vivarium_public_health.risks import RiskEffectSet, get_distribution


class Risk:
    """A model for a risk factor defined by either a continuous or a categorical value. For example,
    (1) high systolic blood pressure as a risk where the SBP is not dichotomized
        into hypotension and normal but is treated as the actual SBP measurement.
    (2) smoking as two categories: current smoker and non-smoker.


       Parameters
       ----------
       risk_type : str
           'risk_factor'
       risk_name : str
           The name of a risk
       """

    def __init__(self, risk_type, risk_name):
        self._risk_type, self._risk = risk_type, risk_name
        self._effects = RiskEffectSet(self._risk, risk_type=self._risk_type)

    def setup(self, builder):
        self.exposure_distribution = get_distribution(self._risk, self._risk_type, builder)
        builder.components.add_components([self._effects, self.exposure_distribution])
        self.randomness = builder.randomness.get_stream(f'initial_{self._risk}_propensity')
        self.population_view = builder.population.get_view(
            [f'{self._risk}_exposure', f'{self._risk}_propensity', 'age', 'sex'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[f'{self._risk}_exposure',
                                                                  f'{self._risk}_propensity'],
                                                 requires_columns=['age', 'sex'])

        builder.event.register_listener('time_step__prepare', self.update_exposure, priority=8)

    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True )
        propensities = pd.Series(self.randomness.get_draw(population.index),
                                 name=f'{self._risk}_propensity',
                                 index=pop_data.index)
        self.population_view.update(propensities)
        exposure = self._get_current_exposure(propensities)
        self.population_view.update(pd.Series(exposure,
                                              name=f'{self._risk}_exposure',
                                              index=pop_data.index))

    def _get_current_exposure(self, propensity):
        return self.exposure_distribution.ppf(propensity)

    def update_exposure(self, event):
        population = self.population_view.get(event.index)
        new_exposure = self._get_current_exposure(population[f'{self._risk}_propensity'])
        self.population_view.update(pd.Series(new_exposure, name=f'{self._risk}_exposure', index=event.index))

    def __repr__(self):
        return f"Risk(_risk_type= {self._risk_type}, _risk= {self._risk})"
