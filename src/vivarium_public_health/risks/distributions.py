"""
=================================
Risk Exposure Distribution Models
=================================

This module contains tools for modeling several different risk
exposure distributions.

"""
import numpy as np
import pandas as pd

from risk_distributions import EnsembleDistribution, Normal, LogNormal

from vivarium.framework.values import list_combiner, joint_value_post_processor


class MissingDataError(Exception):
    pass


class EnsembleSimulation:

    def __init__(self, risk, weights, mean, sd):
        self.risk = risk
        self._weights, self._parameters = self._get_parameters(weights, mean, sd)

    @property
    def name(self):
        return f'ensemble_simulation.{self.risk}'

    def setup(self, builder):
        self.weights = builder.lookup.build_table(self._weights)
        self.parameters = {k: builder.lookup.build_table(v) for k, v in self._parameters.items()}

    def _get_parameters(self, weights, mean, sd):
        index_cols = ['sex', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
        weights = weights.set_index(index_cols)
        mean = mean.set_index(index_cols)['value']
        sd = sd.set_index(index_cols)['value']
        weights, parameters = EnsembleDistribution.get_parameters(weights, mean=mean, sd=sd)
        return weights.reset_index(), {name: p.reset_index() for name, p in parameters.items()}

    def ppf(self, q):
        if not q.empty:
            # We limit valid propensity to [0.001 0.999]. Beyond that bound values return NaN and then become zero,
            # which is nonsensical. We avoid the inclusive limit to protect ourselves from a math error.
            q[q >= 0.999] = 0.998
            q[q <= 0.001] = 0.0011
            weights = self.weights(q.index)
            parameters = {name: parameter(q.index) for name, parameter in self.parameters.items()}
            x = EnsembleDistribution(weights, parameters).ppf(q)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x

    def __repr__(self):
        return f'EnsembleSimulation(risk={self.risk})'


class SimulationDistribution:
    def __init__(self, risk, mean, sd, distribution=None):
        self.risk = risk
        self.distribution = distribution
        self._parameters = self._get_parameters(mean, sd)

    @property
    def name(self):
        return f'simulation_distribution.{self.risk}'

    def setup(self, builder):
        self.parameters = builder.lookup.build_table(self._parameters)

    def _get_parameters(self, mean, sd):
        index = ['sex', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
        mean = mean.set_index(index)['value']
        sd = sd.set_index(index)['value']
        return self.distribution.get_parameters(mean=mean, sd=sd).reset_index()

    def ppf(self, q):
        if not q.empty:
            x = self.distribution(parameters=self.parameters(q.index)).ppf(q)
            x[x.isnull()] = 0
        else:
            x = pd.Series([])
        return x

    def __repr__(self):
        return f"SimulationDistribution(risk={self.risk}, distribution={self.distribution.__name__.lower()})"


class PolytomousDistribution:
    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        self.risk = risk
        self.exposure_data = exposure_data
        self.categories = sorted([column for column in self.exposure_data if 'cat' in column],
                                 key=lambda column: int(column[3:]))

    @property
    def name(self):
        return f'polytomous_distribution.{self.risk}'

    def setup(self, builder):
        self.exposure = builder.value.register_value_producer(f'{self.risk}.exposure_parameters',
                                                              source=builder.lookup.build_table(self.exposure_data))

    def ppf(self, x):
        exposure = self.exposure(x.index)
        sorted_exposures = exposure[self.categories]
        if not np.allclose(1, np.sum(sorted_exposures, axis=1)):
            raise MissingDataError('All exposure data returned as 0.')
        exposure_sum = sorted_exposures.cumsum(axis='columns')
        category_index = (exposure_sum.T < x).T.sum('columns')
        return pd.Series(np.array(self.categories)[category_index], name=self.risk + '_exposure', index=x.index)

    def __repr__(self):
        return f"PolytomousDistribution(risk={self.risk})"


class DichotomousDistribution:
    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        self.risk = risk
        self.exposure_data = exposure_data.drop('cat2', axis=1)

    @property
    def name(self):
        return f'dichotomous_distribution.{self.risk}'

    def setup(self, builder):
        self._base_exposure = builder.lookup.build_table(self.exposure_data)
        self.exposure_proportion = builder.value.register_value_producer(f'{self.risk}.exposure_parameters',
                                                                         source=self.exposure)
        self.joint_paf = builder.value.register_value_producer(f'{self.risk}.exposure_parameters.paf',
                                                               source=lambda index: [builder.lookup.build_table(0)(index)],
                                                               preferred_combiner=list_combiner,
                                                               preferred_post_processor=joint_value_post_processor)

    def exposure(self, index):
        base_exposure = self._base_exposure(index).values
        joint_paf = self.joint_paf(index).values
        return pd.Series(base_exposure * (1-joint_paf), index=index, name='values')

    def ppf(self, x):
        exposed = x < self.exposure_proportion(x.index)
        return pd.Series(exposed.replace({True: 'cat1', False: 'cat2'}), name=self.risk + '_exposure', index=x.index)

    def __repr__(self):
        return f"DichotomousDistribution(risk={self.risk})"


def get_distribution(risk, distribution_type, exposure, exposure_standard_deviation, weights):
    if distribution_type == 'dichotomous':
        distribution = DichotomousDistribution(risk, exposure)
    elif 'polytomous' in distribution_type:
        distribution = PolytomousDistribution(risk, exposure)
    elif distribution_type == 'normal':
        distribution = SimulationDistribution(risk, mean=exposure, sd=exposure_standard_deviation,
                                              distribution=Normal)
    elif distribution_type == 'lognormal':
        distribution = SimulationDistribution(risk, mean=exposure, sd=exposure_standard_deviation,
                                              distribution=LogNormal)
    elif distribution_type == 'ensemble':
        distribution = EnsembleSimulation(risk, weights, mean=exposure, sd=exposure_standard_deviation,)
    else:
        raise NotImplementedError(f"Unhandled distribution type {distribution_type}")
    return distribution

