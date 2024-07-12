"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""

from typing import Any, Dict, List

import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.observer import PublicHealthObserver
from vivarium_public_health.utilities import to_years


class DiseaseObserver(PublicHealthObserver):
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

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.disease_model = builder.components.get_component(f"disease_model.{self.disease}")
        self.entity_type = self.disease_model.cause_type
        self.entity = self.disease_model.cause
        self.transition_stratification_name = f"transition_{self.disease}"

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        return builder.configuration.stratification[self.disease]

    def register_observations(self, builder: Builder) -> None:

        self.register_disease_state_stratification(builder)
        self.register_transition_stratification(builder)

        pop_filter = 'alive == "alive" and tracked==True'
        self.register_person_time_observation(builder, pop_filter)
        self.register_transition_count_observation(builder, pop_filter)

    def register_disease_state_stratification(self, builder: Builder) -> None:
        builder.results.register_stratification(
            self.disease,
            [state.state_id for state in self.disease_model.states],
            requires_columns=[self.disease],
        )

    def register_transition_stratification(self, builder: Builder) -> None:
        transitions = [str(transition) for transition in self.disease_model.transition_names]
        builder.results.register_stratification(
            self.transition_stratification_name,
            categories=transitions + ["no_transition"],
            mapper=self.map_transitions,
            requires_columns=[self.disease, self.previous_state_column_name],
            is_vectorized=True,
        )

    def register_person_time_observation(self, builder: Builder, pop_filter: str) -> None:
        self.register_adding_observation(
            builder=builder,
            name=f"person_time_{self.disease}",
            pop_filter=pop_filter,
            when="time_step__prepare",
            requires_columns=["alive", self.disease],
            additional_stratifications=self.configuration.include + [self.disease],
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_state_person_time,
        )

    def register_transition_count_observation(
        self, builder: Builder, pop_filter: str
    ) -> None:
        self.register_adding_observation(
            builder=builder,
            name=f"transition_count_{self.disease}",
            pop_filter=pop_filter,
            requires_columns=[
                self.previous_state_column_name,
                self.disease,
            ],
            additional_stratifications=self.configuration.include
            + [self.transition_stratification_name],
            excluded_stratifications=self.configuration.exclude,
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

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        results = results.reset_index()
        if "transition_count_" in measure:
            results = results[results[self.transition_stratification_name] != "no_transition"]
            sub_entity = self.transition_stratification_name
        if "person_time_" in measure:
            sub_entity = self.disease
        results.rename(columns={sub_entity: COLUMNS.SUB_ENTITY}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        if "transition_count_" in measure:
            measure_name = "transition_count"
        if "person_time_" in measure:
            measure_name = "person_time"
        return pd.Series(measure_name, index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(self.entity_type, index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series(self.entity, index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        # The sub-entity col was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]
