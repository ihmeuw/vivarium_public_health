"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""

from functools import partial
from typing import Any, Dict, List

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.results import Observer

from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.utilities import to_years


class DiseaseObserver(Observer):
    """Observes disease counts and person time for a cause.

    By default, this observer computes aggregate disease state person time and
    counts of disease events over the full course of the simulation. It can be
    configured to add or remove stratification groups to the default groups
    defined by a ResultsStratifier.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            stratification:
                cause_name:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.disease: super().configuration_defaults["stratification"][
                    self.get_configuration_name()
                ]
            }
        }

    @property
    def columns_created(self) -> List[str]:
        return [self.previous_state_column_name]

    @property
    def columns_required(self) -> List[str]:
        return [self.disease]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [self.disease],
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, disease: str) -> None:
        super().__init__()
        self.disease = disease
        self.previous_state_column_name = f"previous_{self.disease}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.disease]
        self.disease_model = builder.components.get_component(f"disease_model.{self.disease}")

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:

        builder.results.register_stratification(
            self.disease,
            [state.state_id for state in self.disease_model.states],
            requires_columns=[self.disease],
        )

        transition_stratification_name = f"transition_{self.disease}"
        builder.results.register_stratification(
            transition_stratification_name,
            categories=self.disease_model.transition_names + ["no_transition"],
            mapper=self.map_transitions,
            requires_columns=[self.disease, self.previous_state_column_name],
            is_vectorized=True,
        )

        pop_filter = 'alive == "alive" and tracked==True'
        entity_type = self.disease_model.cause_type
        entity = self.disease_model.cause

        builder.results.register_adding_observation(
            name=f"person_time_{self.disease}",
            pop_filter=pop_filter,
            when="time_step__prepare",
            requires_columns=["alive", self.disease],
            results_formatter=partial(
                self.formatter,
                measure_name="person_time",
                entity_type=entity_type,
                entity=entity,
                sub_entity_col=self.disease,
            ),
            additional_stratifications=self.config.include + [self.disease],
            excluded_stratifications=self.config.exclude,
            aggregator=self.aggregate_state_person_time,
        )

        builder.results.register_adding_observation(
            name=f"transition_count_{self.disease}",
            pop_filter=pop_filter,
            when="collect_metrics",
            requires_columns=[
                self.previous_state_column_name,
                self.disease,
            ],
            results_formatter=partial(
                self.formatter,
                measure_name="transition_count",
                entity_type=entity_type,
                entity=entity,
                sub_entity_col=transition_stratification_name,
            ),
            additional_stratifications=self.config.include + [transition_stratification_name],
            excluded_stratifications=self.config.exclude,
        )

    def map_transitions(self, df: pd.DataFrame) -> pd.Series:
        transitions = pd.Series(index=df.index, dtype=str)
        transition_mask = df[self.previous_state_column_name] != df[self.disease]
        transitions[~transition_mask] = "no_transition"
        transitions[transition_mask] = (
            df[self.previous_state_column_name].astype(str)
            + "_to_"
            + df[self.disease].astype(str)
        )
        return transitions

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize the previous state column to the current state"""
        pop = self.population_view.subview([self.disease]).get(pop_data.index)
        pop[self.previous_state_column_name] = pop[self.disease]
        self.population_view.update(pop)

    def on_time_step_prepare(self, event: Event) -> None:
        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column_name] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    ###############
    # Aggregators #
    ###############

    def aggregate_state_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())

    ##################
    # Report methods #
    ##################

    def formatter(
        self,
        measure_name: str,
        entity_type: str,
        entity: str,
        sub_entity_col: str,
        measure: str,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        results = results.reset_index()
        # Remove no_transitions
        if measure_name == "transition_count":
            results = results[results[sub_entity_col] != "no_transition"]
        results.rename(columns={sub_entity_col: COLUMNS.SUB_ENTITY}, inplace=True)
        results[COLUMNS.MEASURE] = measure_name
        results[COLUMNS.ENTITY_TYPE] = entity_type
        results[COLUMNS.ENTITY] = entity

        return results[[c for c in results.columns if c != COLUMNS.VALUE] + [COLUMNS.VALUE]]
