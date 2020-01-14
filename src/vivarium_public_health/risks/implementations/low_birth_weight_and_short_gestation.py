"""
Low birth weight and short gestation (LBWSG) is a non-standard risk
implementation that has been used in several public health models.

Note that because the input data is so large, it relies on a custom relative
risk data loader that expects data saved in keys by draw.
"""
from typing import Tuple
from pathlib import Path

import pandas as pd

from vivarium_public_health.risks import data_transformations as data_transformations
from vivarium_public_health.utilities import EntityString, TargetString
from vivarium_public_health.risks.data_transformations import validate_relative_risk_data_source
from vivarium_public_health.risks.data_transformations import rebin_relative_risk_data
from vivarium_public_health.risks.data_transformations import get_distribution_type
from vivarium_public_health.risks.data_transformations import pivot_categorical
from vivarium_public_health.risks import RiskEffect
from vivarium.framework.randomness import RandomnessStream

MISSING_CATEGORY = 'cat212'


class LBWSGRisk:

    configuration_defaults = {
        'low_birth_weight_and_short_gestation': {
            'exposure': 'data',
            'rebinned_exposed': []
        }
    }

    @property
    def name(self):
        return "low_birth_weight_short_gestation_risk"

    def __init__(self):
        self.risk = EntityString('risk_factor.low_birth_weight_and_short_gestation')

    def setup(self, builder):
        self.exposure_distribution = LBWSGDistribution(builder)

        self.birthweight_gestation_time_view = builder.population.get_view(['birth_weight', 'gestation_time'])

        self._raw_exposure = builder.value.register_value_producer(
            f'{self.risk.name}.raw_exposure',
            source=lambda index: self.birthweight_gestation_time_view.get(index),
            requires_columns=['birth_weight', 'gestation_time']
        )

        self.exposure = builder.value.register_value_producer(
            f'{self.risk.name}.exposure',
            source=self._raw_exposure,
            preferred_post_processor=self.exposure_distribution.convert_to_categorical,
            requires_values=f'{self.risk.name}.raw_exposure'
        )

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=['birth_weight', 'gestation_time'])

    def on_initialize_simulants(self, pop_data):
        exposure = self.exposure_distribution.get_birth_weight_and_gestational_age(pop_data.index)
        self.birthweight_gestation_time_view.update(pd.DataFrame({
            'birth_weight': exposure['birth_weight'],
            'gestation_time': exposure['gestation_time']
        }, index=pop_data.index))


class LBWSGDistribution:

    def __init__(self, builder):
        self.risk = EntityString('risk_factor.low_birth_weight_and_short_gestation')
        self.randomness = builder.randomness.get_stream(f'{self.risk.name}.exposure')

        self.categories_by_interval = get_categories_by_interval(builder, self.risk)
        self.intervals_by_category = self.categories_by_interval.reset_index().set_index('cat')
        self.max_gt_by_bw, self.max_bw_by_gt = self._get_boundary_mappings()

        self.exposure_parameters = builder.lookup.build_table(get_exposure_data(builder, self.risk),
                                                              key_columns=['sex'],
                                                              parameter_columns=['age', 'year'])

    def get_birth_weight_and_gestational_age(self, index):
        category_draw = self.randomness.get_draw(index, additional_key='category')
        exposure = self.exposure_parameters(index)[self.categories_by_interval.values]
        exposure_sum = exposure.cumsum(axis='columns')
        category_index = (exposure_sum.T < category_draw).T.sum('columns')
        categorical_exposure = pd.Series(self.categories_by_interval.values[category_index],
                                         index=index, name='cat')

        return self._convert_to_continuous(categorical_exposure)

    def convert_to_categorical(self, exposure, _):
        exposure = self._convert_boundary_cases(exposure)
        categorical_exposure = self.categories_by_interval.iloc[self._get_categorical_index(exposure)]
        categorical_exposure.index = exposure.index
        return categorical_exposure

    def _convert_boundary_cases(self, exposure):
        eps = 1e-4
        outside_bounds = self._get_categorical_index(exposure) == -1
        shift_down = outside_bounds & (
                (exposure.birth_weight < 1000)
                | ((1000 < exposure.birth_weight) & (exposure.birth_weight < 4500) & (40 < exposure.gestation_time))
        )
        shift_left = outside_bounds & (
                (1000 < exposure.birth_weight) & (exposure.gestation_time < 34)
                | (4500 < exposure.birth_weight) & (exposure.gestation_time < 42)
        )
        tmrel = outside_bounds & (
                (4500 < exposure.birth_weight) & (42 < exposure.gestation_time)
        )

        exposure.loc[shift_down, 'gestation_time'] = (self.max_gt_by_bw
                                                      .loc[exposure.loc[shift_down, 'birth_weight']]
                                                      .values) - eps
        exposure.loc[shift_left, 'birth_weight'] = (self.max_bw_by_gt
                                                    .loc[exposure.loc[shift_left, 'gestation_time']]
                                                    .values) - eps
        exposure.loc[tmrel, 'gestation_time'] = 42 - eps
        exposure.loc[tmrel, 'birth_weight'] = 4500 - eps
        return exposure

    def _get_categorical_index(self, exposure):
        exposure_bw_gt_index = exposure.set_index(['gestation_time', 'birth_weight']).index
        return self.categories_by_interval.index.get_indexer(exposure_bw_gt_index, method=None)

    def _convert_to_continuous(self, categorical_exposure):
        draws = {'birth_weight': self.randomness.get_draw(categorical_exposure.index, additional_key='birth_weight'),
                 'gestation_time': self.randomness.get_draw(categorical_exposure.index, additional_key='gestation_time')}

        def single_values_from_category(row):
            idx = row['index']
            bw_draw = draws['birth_weight'][idx]
            gt_draw = draws['gestation_time'][idx]

            intervals = self.intervals_by_category.loc[row['cat']]

            birth_weight = (intervals.birth_weight.left
                            + bw_draw * (intervals.birth_weight.right - intervals.birth_weight.left))
            gestational_age = (intervals.gestation_time.left
                               + gt_draw * (intervals.gestation_time.right - intervals.gestation_time.left))

            return birth_weight, gestational_age

        values = categorical_exposure.reset_index().apply(single_values_from_category, axis=1)
        return pd.DataFrame(list(values), index=categorical_exposure.index,
                            columns=['birth_weight', 'gestation_time'])

    def _get_boundary_mappings(self):
        cats = self.categories_by_interval.reset_index()
        max_gt_by_bw = pd.Series({bw_interval: pd.Index(group.gestation_time).right.max()
                                  for bw_interval, group in cats.groupby('birth_weight')})
        max_bw_by_gt = pd.Series({gt_interval: pd.Index(group.birth_weight).right.max()
                                  for gt_interval, group in cats.groupby('gestation_time')})
        return max_gt_by_bw, max_bw_by_gt


def get_exposure_data(builder, risk):
    exposure = data_transformations.get_exposure_data(builder, risk)
    exposure[MISSING_CATEGORY] = 0.0
    return exposure


def get_categories_by_interval(builder, risk):
    category_dict = builder.data.load(f'{risk}.categories')
    category_dict[MISSING_CATEGORY] = 'Birth prevalence - [37, 38) wks, [1000, 1500) g'
    cats = (pd.DataFrame.from_dict(category_dict, orient='index')
            .reset_index()
            .rename(columns={'index': 'cat', 0: 'name'}))
    idx = pd.MultiIndex.from_tuples(cats.name.apply(get_intervals_from_name),
                                    names=['gestation_time', 'birth_weight'])
    cats = cats['cat']
    cats.index = idx
    return cats


def get_intervals_from_name(name: str) -> Tuple[pd.Interval, pd.Interval]:
    """Converts a LBWSG category name to a pair of intervals.

    The first interval corresponds to gestational age in weeks, the
    second to birth weight in grams.
    """
    numbers_only = (name.replace('Birth prevalence - [', '')
                    .replace(',', '')
                    .replace(') wks [', ' ')
                    .replace(') g', ''))
    numbers_only = [int(n) for n in numbers_only.split()]
    return (pd.Interval(numbers_only[0], numbers_only[1], closed='left'),
            pd.Interval(numbers_only[2], numbers_only[3], closed='left'))


class LBWSGRiskEffect:
    """A component to model the impact of the low birth weight and short gestation
     risk factor on the target rate of some affected entity.
    """

    configuration_defaults = {
        'effect_of_risk_on_target': {
            'measure': {
                'relative_risk': None,
            }
        }
    }

    @property
    def name(self):
        return f"low_birthweight_short_gestation_risk_effect.{self.target}"

    def __init__(self, target: str):
        """
        Parameters
        ----------
        target :
            Type, name, and target rate of entity to be affected by risk factor,
            supplied in the form "entity_type.entity_name.measure"
            where entity_type should be singular (e.g., cause instead of causes).
        """
        self.risk = EntityString('risk_factor.low_birth_weight_and_short_gestation')
        self.target = TargetString(target)
        self.configuration_defaults = {
            f'effect_of_{self.risk.name}_on_{self.target.name}': {
                self.target.measure: RiskEffect.configuration_defaults['effect_of_risk_on_target']['measure']
            }
        }

    def setup(self, builder):
        self.randomness = builder.randomness.get_stream(f'effect_of_{self.risk.name}_on_{self.target.name}')
        self.relative_risk = builder.lookup.build_table(self.get_relative_risk_data(builder),
                                                        key_columns=['sex'],
                                                        parameter_columns=['age', 'year'])
        self.population_attributable_fraction = builder.lookup.build_table(
            data_transformations.get_population_attributable_fraction_data(builder, self.risk, self.target, self.randomness),
            key_columns=['sex'],
            parameter_columns=['age', 'year']
        )

        self.exposure_effect = data_transformations.get_exposure_effect(builder, self.risk)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}',
                                              modifier=self.adjust_target)
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}.paf',
                                              modifier=self.population_attributable_fraction)

    def adjust_target(self, index, target):
        return self.exposure_effect(target, self.relative_risk(index))

    def get_relative_risk_data(self, builder):
        rr_data = get_relative_risk_data_by_draw(builder, self.risk, self.target, self.randomness)
        rr_data[MISSING_CATEGORY] = (rr_data['cat106'] + rr_data['cat116']) / 2
        return rr_data


# Pulled from vivarium_public_health.risks.data_transformations
def get_relative_risk_data_by_draw(builder, risk: EntityString, target: TargetString, randomness: RandomnessStream):
    source_type = validate_relative_risk_data_source(builder, risk, target)
    relative_risk_data = load_relative_risk_data_by_draw(builder, risk, target, source_type)
    relative_risk_data = rebin_relative_risk_data(builder, risk, relative_risk_data)

    if get_distribution_type(builder, risk) in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        relative_risk_data = pivot_categorical(relative_risk_data)
    else:
        relative_risk_data = relative_risk_data.drop(['parameter'], 'columns')

    return relative_risk_data


def load_relative_risk_data_by_draw(builder, risk: EntityString, target: TargetString, source_type: str):

    artifact_path = Path(builder.data._manager.artifact.path).resolve()

    relative_risk_data = None
    if source_type == 'data':
        relative_risk_data = read_data_by_draw(str(artifact_path), f'{risk}.relative_risk',
                                               builder.configuration.input_data.input_draw_number)
        correct_target = ((relative_risk_data['affected_entity'] == target.name)
                          & (relative_risk_data['affected_measure'] == target.measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))

    return relative_risk_data


def read_data_by_draw(path, key, draw):
    key = key.replace(".", "/")
    with pd.HDFStore(path, mode='r') as store:
        index = store.get(f'{key}/index')
        draw = store.get(f'{key}/draw_{draw}')
    draw.rename("value", inplace=True)
    return pd.concat([index, draw], axis=1)
