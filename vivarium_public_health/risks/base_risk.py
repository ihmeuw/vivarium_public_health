import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from vivarium.framework.randomness import random

from ceam_public_health.risks import RiskEffectSet, get_distribution


def uncorrelated_propensity(population, risk_factor):
    return random(f"initial_propensity_{risk_factor}", population.index)

class ContinuousRiskComponent:
    """A model for a risk factor defined by a continuous value. For example
    high systolic blood pressure as a risk where the SBP is not dichotomized
    into hypotension and normal but is treated as the actual SBP measurement.

    Parameters
    ----------
    risk : str
        The name of a risk factor
    distribution_loader : callable
        A function which take a builder and returns a standard CEAM
        lookup table which returns distribution data.
    exposure_function : callable
        A function which takes the output of the lookup table created
        by distribution_loader and a propensity value for each simulant
        and returns the current exposure to this risk factor.
    """


    def __init__(self, risk_type, risk_name):
        self._risk_type, self._risk = risk_type, risk_name
        self._effects = RiskEffectSet(self._risk, risk_type=self._risk_type)

    def setup(self, builder):

        self.propensity_function = uncorrelated_propensity

        self.exposure_distribution = get_distribution(self._risk, self._risk_type, builder)
        builder.components.add_components([self._effects, self.exposure_distribution])
        self.randomness = builder.randomness.get_stream(self._risk)
        self.population_view = builder.population.get_view(
            [self._risk+'_exposure', self._risk+'_propensity', 'age', 'sex'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self._risk + '_exposure',
                                                                  self._risk + '_propensity'],
                                                 requires_columns=['age', 'sex'])

        builder.event.register_listener('time_step__prepare', self.update_exposure, priority=8)

    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True)
        propensities = pd.Series(self.propensity_function(population, self._risk),
                                 name=self._risk+'_propensity',
                                 index=pop_data.index)
        self.population_view.update(propensities)
        exposure = self._get_current_exposure(propensities)
        self.population_view.update(pd.Series(exposure,
                                              name=self._risk+'_exposure',
                                              index=pop_data.index))

    def _get_current_exposure(self, propensity):
        return self.exposure_distribution.ppf(propensity)

    def update_exposure(self, event):
        population = self.population_view.get(event.index)
        new_exposure = self._get_current_exposure(population[self._risk+'_propensity'])
        self.population_view.update(pd.Series(new_exposure, name=self._risk+'_exposure', index=event.index))

    def __repr__(self):
        return f"ContinuousRiskComponent(_risk_type= {self._risk_type}, _risk= {self._risk})"


class CategoricalRiskComponent:
    """A model for a risk factor defined by a dichotomous value. For example
    smoking as two categories: current smoker and non-smoker.
    Parameters
    ----------
    risk : str
        The name of a risk
    """

    def __init__(self, risk_type, risk_name):
        self._risk_type, self._risk = risk_type, risk_name
        self._effects = RiskEffectSet(self._risk, risk_type=self._risk_type)

    def setup(self, builder):

        self.propensity_function = uncorrelated_propensity

        self.exposure_distribution = get_distribution(self._risk, self._risk_type, builder)
        builder.components.add_components([self._effects, self.exposure_distribution])
        self.randomness = builder.randomness.get_stream(self._risk)
        self.population_view = builder.population.get_view(
            [self._risk+'_propensity', self._risk+'_exposure', 'age', 'sex'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self._risk + '_exposure',
                                                                  self._risk + '_propensity'],
                                                 requires_columns=['age', 'sex'])
        builder.event.register_listener('time_step__prepare', self.update_exposure, priority=8)


    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True)
        propensity =  pd.Series(self.propensity_function(population, self._risk),
                                 name=self._risk+'_propensity',
                                 index=pop_data.index)
        exposure = self._get_current_exposure(propensity)
        self.population_view.update(propensity)
        self.population_view.update(pd.Series(exposure,
                                              name=self._risk + '_exposure',
                                              index=pop_data.index))

    def _get_current_exposure(self, propensity):
        return self.exposure_distribution.ppf(propensity)

    def update_exposure(self, event):
        population = self.population_view.get(event.index)
        new_exposure = self._get_current_exposure(population[self._risk + '_propensity'])
        import pdb; pdb.set_trace()
        self.population_view.update(pd.Series(new_exposure, name=self._risk+'_exposure', index=event.index))

    def __repr__(self):
        return f"CategoricalRiskComponent(_risk_type= {self._risk_type}, _risk= {self._risk})"
