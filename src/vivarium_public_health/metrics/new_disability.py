"""
===================
Disability Observer
===================

This module contains tools for observing years lived with disability (YLDs)
in the simulation.

"""
import typing

import pandas as pd
from vivarium.framework.values import list_combiner, union_post_processor, rescale_post_processor

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.utilities import to_years

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder

# I'm trying to think about how we make these things easier to subclass.
# Perhaps if we are careful we can avoid subclassing by being clever in metrics
# but I think we'll run into too many special cases.
#
# I'm thinking that maybe we can design inheritance simply around overriding
# properties.


class DisabilityObserver:

    @property
    def name(self):
        return 'disability_observer'

    @property
    def disease_classes(self):
        return [DiseaseState, RiskAttributableDisease]

    @property
    def additional_groupers(self):
        return []

    @property
    def additional_columns(self):
        return []

    def setup(self, builder: 'Builder'):
        self.disability_weight = builder.value.register_value_producer(
            'disability_weight',
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=_disability_post_processor)

        cause_states = builder.components.get_components_by_type(tuple(self.disease_classes))
        cause_names = [c.state_id for c in cause_states]
        self.disability_weight_pipelines = {cause: builder.value.get_value(f'{cause}.disability_weight')
                                            for cause in cause_names}

        params = builder.configuration.metrics.stratification_params
        groupers = [] + self.additional_groupers
        columns = ['tracked', 'alive'] + self.additional_columns
        if 'age_group' in params:
            groupers.append('age_group')
            columns.append('age')
        if 'sex' in params:
            groupers.append('sex')
            columns.append('sex')
        if 'year' in params:
            groupers.append('current_year')

        for cause in cause_states:
            name = cause.state_id
            pipeline = builder.value.get_value(f'{name}.disability_weight')
            builder.results.register_results_producer(
                measure=f'ylds_due_to_{name}',
                pop_filter='alive == "alive"',
                groupers=groupers,
                aggregator=_dw_aggregator(pipeline),
                requires_columns=columns,
                when='time_step__prepare',
            )
        builder.results.register_results_producer(
            measure='ylds_due_to_all_causes',
            pop_filter='alive == "alive"',
            groupers=groupers,
            aggregator=_dw_aggregator(self.disability_weight),
            requires_columns=columns,
            when='time_step__prepare',
        )

    def __repr__(self):
        return "DisabilityObserver()"


def _disability_post_processor(value, step_size):
    return rescale_post_processor(union_post_processor(value, step_size), step_size)


def _dw_aggregator(pipeline):
    def _aggregate(group):
        return (pipeline(group.index) * to_years(group['step_size'])).sum()
    return _aggregate
