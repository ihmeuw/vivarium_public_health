import numpy as np
import pandas as pd

from risk_distributions import risk_distributions

from vivarium.framework.values import list_combiner, joint_value_post_processor
from vivarium_public_health.risks.data_transformations import pivot_categorical

class MissingDataError(Exception):
    pass


class EnsembleSimulation:
    def __init__(self, weights, mean, sd):
        self._weights = pivot_categorical(weights)
        self._mean = mean
        self._sd = sd

    def setup(self, builder):
        self.weights = builder.lookup.build_table(self._weights)
        self.mean = builder.lookup.build_table(self._mean.drop('parameter', axis=1))
        self.sd = builder.lookup.build_table(self._sd)

    def ppf(self, x):
        ensemble = risk_distributions.EnsembleDistribution(self.weights(x.index), self.mean(x.index), self.sd(x.index))
        return ensemble.ppf(x)


class SimulationDistribution:
    def __init__(self, mean, sd, distribution=None):
        self.distribution = distribution
        self._parameters = self._get_parameters(mean, sd)

    def setup(self, builder):
        self.parameters = builder.lookup.build_table(self._parameters)

    def _get_parameters(self, mean, sd):
        index = ['sex', 'age_group_start', 'age_group_end', 'year_start', 'year_end']
        mean = mean.set_index(index)['value']
        sd = sd.set_index(index)['value']
        assert mean[mean == 0].index.difference(sd[sd == 0].index).empty
        params = self.distribution.get_params(mean[mean > 0], sd[sd > 0]).set_index(mean[mean > 0].index)
        params = params.reindex(mean.index, fill_value=0).reset_index()
        return params

    def ppf(self, x):
        return self.distribution(params=self.parameters(x.index)).ppf(x)


class PolytomousDistribution:
    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        self.risk = risk
        self.exposure_data = exposure_data
        self.categories = sorted([column for column in self.exposure_data if 'cat' in column],
                                 key=lambda column: int(column[3:]))

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


class DichotomousDistribution:
    def __init__(self, risk: str, exposure_data: pd.DataFrame):
        self.risk = risk
        self.exposure_data = exposure_data.drop('cat2', axis=1)

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


def get_distribution(risk, distribution_type, exposure, exposure_standard_deviation, weights):
    if distribution_type == 'dichotomous':
        distribution = DichotomousDistribution(risk, exposure)
    elif 'polytomous' in distribution_type:
        distribution = PolytomousDistribution(risk, exposure)
    elif distribution_type == 'normal':
        distribution = SimulationDistribution(mean=exposure, sd=exposure_standard_deviation,
                                              distribution=risk_distributions.Normal)
    elif distribution_type == 'lognormal':
        distribution = SimulationDistribution(mean=exposure, sd=exposure_standard_deviation,
                                              distribution=risk_distributions.LogNormal)
    elif distribution_type == 'ensemble':
        distribution = EnsembleSimulation(weights, mean=exposure, sd=exposure_standard_deviation,)
    else:
        raise NotImplementedError(f"Unhandled distribution type {distribution_type}")
    return distribution
