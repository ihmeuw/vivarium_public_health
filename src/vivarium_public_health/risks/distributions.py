import numpy as np
import pandas as pd

from risk_distributions import risk_distributions

from vivarium.framework.values import list_combiner, joint_value_post_processor
from functools import partial


class MissingDataError(Exception):
    pass


class EnsembleSimulation(risk_distributions.EnsembleDistribution):

    def setup(self, builder):
        builder.components.add_components(self._distributions.values())

    def get_distribution_map(self):
        dist_map = super().get_distribution_map()
        return {dist_name: partial(SimulationDistribution, distribution=dist) for dist_name, dist in dist_map.items()}


class SimulationDistribution:
    def __init__(self, mean, sd, distribution=None):
        self.distribution = distribution
        self._parameters = distribution.get_params(mean, sd)

    def setup(self, builder):
        self.parameters = builder.lookup.build_table(self._parameters.reset_index())

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
        return base_exposure * (1-joint_paf)

    def ppf(self, x):
        exposed = x < self.exposure_proportion(x.index)
        return pd.Series(exposed.replace({True: 'cat1', False: 'cat2'}), name=self.risk + '_exposure', index=x.index)


def get_distribution(risk, distribution_type, exposure, exposure_standard_deviation, weights):
    if distribution_type == 'dichotomous':
        distribution = DichotomousDistribution(exposure, risk)
    elif 'polytomous' in distribution_type:
        distribution = PolytomousDistribution(exposure, risk)
    elif distribution_type == 'normal':
        distribution = SimulationDistribution(mean=exposure['value'], sd=exposure_standard_deviation['value'],
                                              distribution=risk_distributions.Normal)
    elif distribution_type == 'lognormal':
        distribution = SimulationDistribution(mean=exposure['value'], sd=exposure_standard_deviation['value'],
                                              distribution=risk_distributions.LogNormal)
    elif distribution_type == 'ensemble':
        # weight is all same across the demographic groups
        e_weights = weights.head(1)
        distribution = EnsembleSimulation(e_weights, mean=exposure['value'], sd=exposure_standard_deviation['value'],)
    else:
        raise NotImplementedError(f"Unhandled distribution type {distribution_type}")
    return distribution
