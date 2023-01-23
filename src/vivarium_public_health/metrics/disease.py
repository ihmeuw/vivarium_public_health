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

        disease_model = builder.components.get_component(f"disease_model.{self.disease}")
        self.previous_state_column = f"previous_{self.disease}"

        builder.population.initializes_simulants(
            self.on_initialize_simulants, creates_columns=[self.previous_state_column]
        )

        columns_required = [self.disease, self.previous_state_column]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)

        for state in disease_model.states:
            builder.results.register_observation(
                name=f"{state.state_id}_person_time",
                pop_filter=f'alive == "alive" and {self.disease} == "{state.state_id}"',
                aggregator=self.aggregate_state_person_time,
                requires_columns=["alive", self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )

        for transition in disease_model.transition_names:
            filter_string = (
                f'{self.previous_state_column} == "{transition.from_state}" '
                f'and {self.disease} == "{transition.to_state}"'
            )
            builder.results.register_observation(
                name=f"{transition}_event_count",
                pop_filter=filter_string,
                requires_columns=[self.previous_state_column, self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="collect_metrics",
            )

    def aggregate_state_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(
            pd.Series("", index=pop_data.index, name=self.previous_state_column)
        )

    def on_time_step_prepare(self, event) -> None:
        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)
