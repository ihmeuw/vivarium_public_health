"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""
from collections import Counter
from typing import Dict, List

import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_public_health.utilities import to_years


class DiseaseObserver:
    configuration_defaults = {
        "stratification": {
        "disease": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        # self.configuration_defaults = self._get_configuration_defaults()
        #XXX TODO fix this!
        self.configuration_defaults = {
        "stratification": {
            "disease": {
                "exclude": ["age_group"],
                "include": ["sex"],
            }
        }
    }

        self.disease_component_name = f"disease_model.{self.disease}"
        self.current_state_column_name = self.disease
        self.previous_state_column_name = f"previous_{self.disease}"
        self.metrics_pipeline_name = "metrics"

    def __repr__(self):
        return f"DiseaseObserver({self.disease})"

    @property
    def name(self):
        return f"disease_observer_{self.disease}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.disease]

        disease_model = builder.components.get_component(f'disease_model.{self.disease}')
        self.previous_state_column = f'previous_{self.disease}'

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.previous_state_column]
        )

        columns_required = [f'{self.disease}', self.previous_state_column]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

        for state in disease_model.states:
            builder.results.register_observation(
                name=f'{state}_person_time',
                pop_filter=f'alive == "alive" and {self.disease} == "{state}"',
                aggregator=self.aggregate_state_person_time,
                requires_columns=["alive", self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when='time_step__prepare',
            )

        for transition in disease_model.transition_names:
            filter = (
                f'{self.previous_state_column} == {transition.from_state} '
                f'and {self.disease} == {transition.to_state}'
            )
            builder.results.register_observation(
                name=f'{transition}_event_count',
                pop_filter=filter,
                requires_columns=[self.previous_state_column, self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when='collect_metrics',
            )

    def aggregate_state_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())

    def _get_configuration_defaults(self) -> Dict[str, Dict]:  # XXX TODO: test this, prolly isn't right
        return {
            "observers": {
                self.disease: DiseaseObserver.configuration_defaults["stratification"]["disease"]
            }
        }

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series('', index=pop_data.index, name=self.previous_state_column))

    def on_time_step_prepare(self, event) -> None:
        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

#
# class DiseaseObserver:
#     """Observes disease counts and person time for a cause.
#
#     By default, this observer computes aggregate susceptible person time and
#     counts of disease cases over the full course of the simulation. It can be
#     configured to add or remove stratification groups to the default groups
#     defined by a ResultsStratifier.
#
#     In the model specification, your configuration for this component should
#     be specified as, e.g.:
#
#     .. code-block:: yaml
#
#         configuration:
#             observers:
#                 cause_name:
#                     exclude:
#                         - "sex"
#                     include:
#                         - "sample_stratification"
#
#     """
#
#     configuration_defaults = {
#         "observers": {
#             "disease": {
#                 "exclude": [],
#                 "include": [],
#             }
#         }
#     }
#
#     def __init__(self, disease: str):
#         self.disease = disease
#         self.configuration_defaults = self._get_configuration_defaults()
#
#         self.disease_component_name = f"disease_model.{self.disease}"
#         self.current_state_column_name = self.disease
#         self.previous_state_column_name = f"previous_{self.disease}"
#         self.metrics_pipeline_name = "metrics"
#
#     def __repr__(self):
#         return f"DiseaseObserver({self.disease})"
#
#     ##########################
#     # Initialization methods #
#     ##########################
#
#     # noinspection PyMethodMayBeStatic
#     def _get_configuration_defaults(self) -> Dict[str, Dict]:
#         return {
#             "observers": {
#                 self.disease: DiseaseObserver.configuration_defaults["observers"]["disease"]
#             }
#         }
#
#     ##############
#     # Properties #
#     ##############
#
#     @property
#     def name(self):
#         return f"disease_observer.{self.disease}"
#
#     #################
#     # Setup methods #
#     #################
#
#     # noinspection PyAttributeOutsideInit
#     def setup(self, builder: Builder) -> None:
#         self.config = self._get_stratification_configuration(builder)
#         self.stratifier = self._get_stratifier(builder)
#         self.states = self._get_states(builder)
#         self.transitions = self._get_transitions(builder)
#         self.population_view = self._get_population_view(builder)
#
#         self.counts = Counter()
#
#         self._register_simulant_initializer(builder)
#         self._register_time_step_prepare_listener(builder)
#         self._register_collect_metrics_listener(builder)
#         self._register_metrics_modifier(builder)
#
#     def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
#         return builder.configuration.observers[self.disease]
#
#     # noinspection PyMethodMayBeStatic
#     def _get_stratifier(self, builder: Builder) -> ResultsStratifier:
#         return builder.components.get_component(ResultsStratifier.name)
#
#     def _get_states(self, builder: Builder) -> List[str]:
#         return builder.components.get_component(self.disease_component_name).state_names
#
#     def _get_transitions(self, builder: Builder) -> List[TransitionString]:
#         return builder.components.get_component(self.disease_component_name).transition_names
#
#     # noinspection PyMethodMayBeStatic
#     def _get_population_view(self, builder: Builder) -> PopulationView:
#         columns_required = [
#             self.current_state_column_name,
#             self.previous_state_column_name,
#         ]
#         return builder.population.get_view(columns_required)
#
#     def _register_simulant_initializer(self, builder: Builder) -> None:
#         # todo observer should not be modifying state table
#         builder.population.initializes_simulants(
#             self.on_initialize_simulants,
#             creates_columns=[self.previous_state_column_name],
#         )
#
#     def _register_time_step_prepare_listener(self, builder: Builder) -> None:
#         # In order to get an accurate representation of person time we need to look at
#         # the state table before anything happens.
#         builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
#
#     def _register_collect_metrics_listener(self, builder: Builder) -> None:
#         builder.event.register_listener("collect_metrics", self.on_collect_metrics)
#
#     def _register_metrics_modifier(self, builder: Builder) -> None:
#         builder.value.register_value_modifier(
#             self.metrics_pipeline_name,
#             modifier=self.metrics,
#         )
#
#     ########################
#     # Event-driven methods #
#     ########################
#
#     def on_initialize_simulants(self, pop_data: pd.DataFrame) -> None:
#         self.population_view.update(
#             pd.Series("", index=pop_data.index, name=self.previous_state_column_name)
#         )
#
#     def on_time_step_prepare(self, event: Event) -> None:
#         step_size_in_years = to_years(event.step_size)
#         pop = self.population_view.get(
#             event.index, query='tracked == True and alive == "alive"'
#         )
#         groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
#         for label, group_mask in groups:
#             for state in self.states:
#                 state_in_group_mask = group_mask & (
#                     pop[self.current_state_column_name] == state
#                 )
#                 person_time_in_group = state_in_group_mask.sum() * step_size_in_years
#                 new_observations = {
#                     f"{self.disease}_{state}_person_time_{label}": person_time_in_group
#                 }
#                 self.counts.update(new_observations)
#
#         # todo observer should not be maintaining this column
#         # This enables tracking of transitions between states
#         pop[self.previous_state_column_name] = pop[self.disease]
#         self.population_view.update(pop)
#
#     def on_collect_metrics(self, event: Event) -> None:
#         pop = self.population_view.get(
#             event.index, query='tracked == True and alive == "alive"'
#         )
#         groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
#         for label, group_mask in groups:
#             for transition in self.transitions:
#                 transition_mask = (
#                     group_mask
#                     & (pop[self.previous_state_column_name] == transition.from_state)
#                     & (pop[self.current_state_column_name] == transition.to_state)
#                 )
#                 new_observations = {
#                     f"{self.disease}_{transition}_event_count_{label}": transition_mask.sum()
#                 }
#                 self.counts.update(new_observations)
#
#     ##################################
#     # Pipeline sources and modifiers #
#     ##################################
#
#     # noinspection PyUnusedLocal
#     def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
#         metrics.update(self.counts)
#         return metrics
