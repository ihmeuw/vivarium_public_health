"""
===================
Disability Observer
===================
This module contains tools for observing years lived with disability (YLDs)
in the simulation.
"""
import typing
from typing import List

import pandas as pd
from vivarium.framework.values import list_combiner, union_post_processor, rescale_post_processor

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.utilities import to_years

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class DisabilityObserver:

    @property
    def name(self) -> str:
        return 'disability_observer'

    @property
    def disease_classes(self) -> List:
        return [DiseaseState, RiskAttributableDisease]

    def setup(self, builder: 'Builder') -> None:
        self.disability_weight = builder.value.register_value_producer(
            'disability_weight',
            source=lambda index: [pd.Series(0.0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=_disability_post_processor
        )
        builder.results.add_results_production_strategy(
            measure='ylds_due_to_all_causes',
            pop_filter='tracked == True and alive == "alive"',
            aggregator=_dw_aggregator(self.disability_weight),
            when='time_step__prepare',
        )
        cause_states = builder.components.get_components_by_type(tuple(self.disease_classes))
        for cause_state in cause_states:
            pipeline = builder.value.get_value(f'{cause_state.state_id}.disability_weight')
            builder.results.add_results_production_strategy(
                measure=f'ylds_due_to_{cause_state.state_id}',
                pop_filter='alive == "alive"',
                aggregator=_dw_aggregator(pipeline),
                when='time_step__prepare',
            )

    def __repr__(self) -> str:
        return "DisabilityObserver()"


def _disability_post_processor(value, step_size):
    return rescale_post_processor(union_post_processor(value, step_size), step_size)


def _dw_aggregator(pipeline):
    def _aggregate(group):
        return (pipeline(group.index) * to_years(group['step_size'])).sum()
    return _aggregate
