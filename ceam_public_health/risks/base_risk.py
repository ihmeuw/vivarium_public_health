import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from ceam_inputs import (risk_factors, coverage_gaps, get_risk_correlation_matrix,
                         get_exposure, get_exposure_standard_deviation)

from vivarium.framework.randomness import random

from ceam_public_health.risks import make_gbd_risk_effects, get_distribution


def uncorrelated_propensity(population, risk_factor):
    return random('initial_propensity_{}'.format(risk_factor.name), population.index)


def correlated_propensity_factory(config):

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
        correlation_matrices = get_risk_correlation_matrix(override_config=config)

        if correlation_matrices is None or risk_factor.name not in correlation_matrices.risk_factor.unique():
            # There's no correlation data for this risk, just pick a uniform random propensity
            return uncorrelated_propensity(population, risk_factor)

        correlation_matrices = correlation_matrices.set_index(
            ['risk_factor', 'sex', 'age']).sort_index(0).sort_index(1).reset_index()

        risk_factor_idx = sorted(correlation_matrices.risk_factor.unique()).index(risk_factor.name)
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
            draw = dist.rvs(group.index.max()+1, random_state=config.run_configuration.input_draw_number)
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
    risk : ceam_inputs.gbd_mapping.risk_factors element
        The configuration data for the risk
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
            'apply_correlation': True,
        },
    }

    def __init__(self, risk, propensity_function=None):
        self._risk = risk_factors[risk] if isinstance(risk, str) else risk
        self._effects = make_gbd_risk_effects(self._risk)
        self.propensity_function = propensity_function

    def setup(self, builder):
        if self.propensity_function is None:
            if builder.configuration.risks.apply_correlation:
                self.propensity_function = correlated_propensity_factory(builder.configuration)
            else:
                self.propensity_function = uncorrelated_propensity

        exposure_data = get_exposure(self._risk, builder.configuration)
        exposure_sd_data = get_exposure_standard_deviation(self._risk, builder.configuration)
        exposure = exposure_data.merge(exposure_sd_data).set_index(['age', 'sex', 'year'])

        self.exposure_distribution = get_distribution(self._risk, exposure)
        self.randomness = builder.randomness.get_stream(self._risk.name)
        self.population_view = builder.population.get_view(
            [self._risk.name+'_exposure', self._risk.name+'_propensity', 'age', 'sex'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self._risk.name + '_exposure',
                                                                  self._risk.name + '_propensity'],
                                                 requires_columns=['age', 'sex'])

        builder.event.register_listener('time_step__prepare', self.update_exposure, priority=8)

        return self._effects + [self.exposure_distribution]

    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True)
        propensities = pd.Series(self.propensity_function(population, self._risk),
                                 name=self._risk.name+'_propensity',
                                 index=pop_data.index)
        self.population_view.update(propensities)
        exposure = self._get_current_exposure(propensities)
        self.population_view.update(pd.Series(exposure,
                                              name=self._risk.name+'_exposure',
                                              index=pop_data.index))


    def _get_current_exposure(self, propensity):
        return self.exposure_distribution.ppf(propensity)

    def update_exposure(self, event):
        population = self.population_view.get(event.index)
        new_exposure = self._get_current_exposure(population[self._risk.name+'_propensity'])
        self.population_view.update(pd.Series(new_exposure, name=self._risk.name+'_exposure', index=event.index))

    def __repr__(self):
        return "ContinuousRiskComponent(_risk= {})".format(self._risk.name)


class CategoricalRiskComponent:
    """A model for a risk factor defined by a dichotomous value. For example
    smoking as two categories: current smoker and non-smoker.
    Parameters
    ----------
    risk : ceam_inputs.gbd_mapping.risk_factors element
        The configuration data for the risk
    """

    configuration_defaults = {
        'risks': {
            'apply_correlation': True,
        },
    }

    def __init__(self, risk, propensity_function=None):
        if isinstance(risk, str):
            self._risk = risk_factors[risk] if risk in risk_factors else coverage_gaps[risk]
        else:
            self._risk = risk
        self._effects = make_gbd_risk_effects(self._risk)
        self.propensity_function = propensity_function

    def setup(self, builder):
        if self.propensity_function is None:
            if builder.configuration.risks.apply_correlation:
                self.propensity_function = correlated_propensity_factory(builder.configuration)
            else:
                self.propensity_function = uncorrelated_propensity

        self.population_view = builder.population.get_view(
            [self._risk.name+'_propensity', self._risk.name+'_exposure', 'age', 'sex'])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self._risk.name + '_exposure',
                                                                  self._risk.name + '_propensity'],
                                                 requires_columns=['age', 'sex'])

        exposure_data = get_exposure(risk=self._risk, override_config=builder.configuration)
        exposure_data = pd.pivot_table(exposure_data, index=['year', 'age', 'sex'], columns='parameter', values='mean')
        exposure_data = exposure_data.reset_index()

        self.exposure = builder.value.register_value_producer(f'{self._risk.name}.exposure',
                                                              source=builder.lookup(exposure_data))

        self.randomness = builder.randomness.get_stream(self._risk.name)
        builder.event.register_listener('time_step__prepare', self.update_exposure, priority=8)

        return self._effects

    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True)
        propensity = self.propensity_function(population, self._risk)
        exposure = self._get_current_exposure(propensity)
        self.population_view.update(pd.DataFrame({
            self._risk.name+'_propensity': propensity,
            self._risk.name+'_exposure': exposure,
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

        return pd.Series(np.array(categories)[category_index], name=self._risk.name+'_exposure')

    def update_exposure(self, event):
        pop = self.population_view.get(event.index)

        propensity = pop[self._risk.name+'_propensity']
        categories = self._get_current_exposure(propensity)
        self.population_view.update(categories)

    def __repr__(self):
        return "CategoricalRiskComponent(_risk= {})".format(self._risk.name)
