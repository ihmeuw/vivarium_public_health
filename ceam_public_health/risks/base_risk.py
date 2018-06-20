import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from vivarium.framework.randomness import random

from ceam_public_health.risks import RiskEffectSet, get_distribution


def uncorrelated_propensity(population, risk_factor):
    return random(f"initial_propensity_{risk_factor}", population.index)


def correlated_propensity_factory(builder):
    input_draw_number = builder.configuration.run_configuration.input_draw_number
    correlation_matrices_data = builder.data.load("risk_factor.correlations.correlations")

    def correlated_propensity(population, risk_factor):
        """Choose a propensity to the risk factor for each simulant that respects
        the risk factor's expected correlation with other risk factors unless there
        is no correlation data available in which case a uniformly distributed
        random propensity will be chosen.

        Parameters
        ----------
        population: pd.DataFrame
            The population to get propensities for. Must include 'sex' and 'age' columns.
        risk_factor: `vivarium.config_tree.ConfigTree`
            The gbd data mapping for the risk.

        Notes
        -----
        This function does some calculation, including loading the correlation
        matrices, once for each risk where they could be done once for all
        risks. In practice this doesn't seem to cause meaningful performance
        problems and it simplifies the code significantly. It's something to be aware
        of though, especially for a model with high fertility or migration where
        this code may be run in each time step rather than primarily during
        initialization.
        """

        correlation_matrices = correlation_matrices_data.set_index(
            ['risk_factor', 'sex', 'age']).sort_index(0).sort_index(1).reset_index()

        if correlation_matrices is None or risk_factor not in correlation_matrices.risk_factor.unique():
            # There's no correlation data for this risk, just pick a uniform random propensity
            return uncorrelated_propensity(population, risk_factor)

        risk_factor_idx = sorted(correlation_matrices.risk_factor.unique()).index(risk_factor)
        ages = sorted(correlation_matrices.age.unique())
        age_idx = (np.broadcast_to(population.age, (len(ages), len(population))).T
                   >= np.broadcast_to(ages, (len(population), len(ages))))
        age_idx = np.minimum(len(ages) - 1, np.sum(age_idx, axis=1))

        qdist = norm(loc=0, scale=1)
        quantiles = pd.Series()

        for (sex, age_idx), group in population.groupby((population.sex, age_idx)):
            matrix = correlation_matrices.query('age == @ages[@age_idx] and sex == @sex')
            del matrix['age']
            del matrix['sex']
            matrix = matrix.set_index(['risk_factor'])
            matrix = matrix.values

            dist = multivariate_normal(mean=np.zeros(len(matrix)), cov=matrix)
            draw = dist.rvs(group.index.max()+1, random_state=input_draw_number)
            draw = draw[group.index]
            quantiles = quantiles.append(
                pd.Series(qdist.cdf(draw).T[risk_factor_idx], index=group.index)
            )

        return pd.Series(quantiles, index=population.index)
    return correlated_propensity


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

    configuration_defaults = {
        'risks': {
            'apply_correlation': False,
        },
    }

    def __init__(self, risk):
        self._risk_type, self._risk = risk.split('.')
        self._effects = RiskEffectSet(self._risk, risk_type=self._risk_type)

    def setup(self, builder):

        if builder.configuration.risks.apply_correlation:
            self.propensity_function = correlated_propensity_factory(builder)
        else:
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
        return f"ContinuousRiskComponent(_risk= {self._risk})"


class CategoricalRiskComponent:
    """A model for a risk factor defined by a dichotomous value. For example
    smoking as two categories: current smoker and non-smoker.
    Parameters
    ----------
    risk : str
        The name of a risk
    """

    configuration_defaults = {
        'risks': {
            'apply_correlation': False,
        },
    }

    def __init__(self, risk):
        self._risk_type, self._risk = risk.split('.')
        self._effects = RiskEffectSet(self._risk, risk_type=self._risk_type)

    def setup(self, builder):
        builder.components.add_components([self._effects])
        if builder.configuration.risks.apply_correlation:
            self.propensity_function = correlated_propensity_factory(builder)
        else:
            self.propensity_function = uncorrelated_propensity

        self.population_view = builder.population.get_view(
            [self._risk+'_propensity', self._risk+'_exposure', 'age', 'sex'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self._risk + '_exposure',
                                                                  self._risk + '_propensity'],
                                                 requires_columns=['age', 'sex'])

        exposure_data = builder.data.load(f"{self._risk_type}.{self._risk}.exposure")
        exposure_data = pd.pivot_table(exposure_data,
                                       index=['year', 'age', 'sex'],
                                       columns='parameter', values='value'
                                      ).dropna().reset_index()

        self.exposure = builder.value.register_value_producer(f'{self._risk}.exposure',
                                                              source=builder.lookup.build_table(exposure_data))

        self.randomness = builder.randomness.get_stream(self._risk)
        builder.event.register_listener('time_step__prepare', self.update_exposure, priority=8)

    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True)
        propensity = self.propensity_function(population, self._risk)
        exposure = self._get_current_exposure(propensity)
        self.population_view.update(pd.DataFrame({
            self._risk+'_propensity': propensity,
            self._risk+'_exposure': exposure,
        }))

    def _get_current_exposure(self, propensity):
        exposure = self.exposure(propensity.index)

        # Get a list of sorted category names (e.g. ['cat1', 'cat2', ..., 'cat9', 'cat10', ...])
        categories = sorted([column for column in exposure if 'cat' in column])
        sorted_exposures = exposure[categories]
        exposure_sum = sorted_exposures.cumsum(axis='columns')
        # Sometimes all data is 0 for the category exposures.  Set the "no exposure" category to catch this case.
        exposure_sum[categories[-1]] = 1  # TODO: Something better than this.

        category_index = (exposure_sum.T < propensity).T.sum('columns')

        return pd.Series(np.array(categories)[category_index], name=self._risk+'_exposure', index=propensity.index)

    def update_exposure(self, event):
        pop = self.population_view.get(event.index)

        propensity = pop[self._risk+'_propensity']
        categories = self._get_current_exposure(propensity)
        self.population_view.update(categories)

    def __repr__(self):
        return f"CategoricalRiskComponent(_risk= {self._risk})"
