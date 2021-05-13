"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""
import typing

import pandas as pd

from vivarium_public_health.utilities import to_years

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class TransitionString(str):

    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split('_TO_')
        return obj


class DiseaseObserver:

    def __init__(self, disease: str) -> None:
        self.disease = disease

    @property
    def name(self) -> str:
        return f'disease_observer.{self.disease}'

    def setup(self, builder: 'Builder') -> None:
        self.step_size = builder.time.step_size()

        self.previous_state_column = f'previous_{self.disease}'
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.previous_state_column])

        columns_required = ['alive', f'{self.disease}', self.previous_state_column]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

        disease_model = builder.components.get_component(f'disease_model.{self.disease}')
        for state in disease_model.states:
            builder.results.add_results_production_strategy(
                measure=f'{state}_person_time',
                pop_filter=f'alive == "alive" and {self.disease} == "{state}"',
                aggregator=self.aggregate_state_person_time,
                when='time_step__prepare',
            )

        for transition in disease_model.transitions:
            transition = TransitionString(transition)
            builder.results.add_results_production_strategy(
                measure=f'{transition}_event_count',
                pop_filter=(f'{self.previous_state_column} == {transition.from_state} '
                            f'and {self.disease} == {transition.to_state}'),
                when='collect_metrics',
            )

    def aggregate_state_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series('', index=pop_data.index, name=self.previous_state_column))

    def on_time_step_prepare(self, event) -> None:
        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    def __repr__(self) -> str:
        return "DiseaseObserver()"
