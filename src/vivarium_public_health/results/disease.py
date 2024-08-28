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

    Attributes
    ----------
    disease
        The name of the disease being observed.
    previous_state_column_name
        The name of the column that stores the previous state of the disease.
    step_size
        The time step size of the simulation.
    disease_model
        The disease model for the disease being observed.
    entity_type
        The type of entity being observed.
    entity
        The entity being observed.
    transition_stratification_name
        The stratification name for transitions between disease states.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return {
            "stratification": {
                self.disease: super().configuration_defaults["stratification"][
                    self.get_configuration_name()
                ]
            }
        }

    @property
    def columns_created(self) -> List[str]:
        """Columns created by this observer."""
        return [self.previous_state_column_name]

    @property
    def columns_required(self) -> List[str]:
        """Columns required by this observer."""
        return [self.disease]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        """Requirements for observer initialization."""
        return {
            "requires_columns": [self.disease],
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, disease: str) -> None:
        """Constructor for this observer.

        Parameters
        ----------
        disease
            The name of the disease being observed.
        """
        super().__init__()
        self.disease = disease
        self.previous_state_column_name = f"previous_{self.disease}"

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Set up the observer."""
        self.step_size = builder.time.step_size()
        self.disease_model = builder.components.get_component(f"disease_model.{self.disease}")
        self.entity_type = self.disease_model.cause_type
        self.entity = self.disease_model.cause
        self.transition_stratification_name = f"transition_{self.disease}"

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        """Get the stratification configuration for this observer.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The stratification configuration for this observer.
        """
        return builder.configuration.stratification[self.disease]

    def register_observations(self, builder: Builder) -> None:
        """Register stratifications and observations.

        Notes
        -----
        Ideally, each observer registers a single observation. This one, however,
        registeres two.

        While it's typical for all stratification registrations to be encapsulated
        in a single class (i.e. the
        :class:ResultsStratifier <vivarium_public_health.results.stratification.ResultsStratifier),
        this observer registers two additional stratifications. While they could
        be registered in the ``ResultsStratifier`` as well, they are specific to
        this observer and so they are registered here while we have easy access
        to the required names and categories.
        """
        self.register_disease_state_stratification(builder)
        self.register_transition_stratification(builder)

        pop_filter = 'alive == "alive" and tracked==True'
        self.register_person_time_observation(builder, pop_filter)
        self.register_transition_count_observation(builder, pop_filter)

    def register_disease_state_stratification(self, builder: Builder) -> None:
        """Register the disease state stratification."""
        builder.results.register_stratification(
            self.disease,
            [state.state_id for state in self.disease_model.states],
            requires_columns=[self.disease],
        )

    def register_transition_stratification(self, builder: Builder) -> None:
        """Register the transition stratification.

        This stratification is used to track transitions between disease states.
        It appends 'no_transition' to the list of transition categories and also
        includes it as an exluded category.

        Notes
        -----
        It is important to include 'no_transition' in bith the list of transition
        categories as well as the list of excluded categories. This is because
        it must exist as a category for the transition mapping to work correctly,
        but then we don't want to include it later during the actual stratification
        process.
        """
        transitions = [
            str(transition) for transition in self.disease_model.transition_names
        ] + ["no_transition"]
        # manually append 'no_transition' as an excluded transition
        excluded_categories = (
            builder.configuration.stratification.excluded_categories.to_dict().get(
                self.transition_stratification_name, []
            )
        ) + ["no_transition"]
        builder.results.register_stratification(
            self.transition_stratification_name,
            categories=transitions,
            excluded_categories=excluded_categories,
            mapper=self.map_transitions,
            requires_columns=[self.disease, self.previous_state_column_name],
            is_vectorized=True,
        )

    def register_person_time_observation(self, builder: Builder, pop_filter: str) -> None:
        """Register a person time observation."""
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
        """Register a transition count observation."""
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
        """Map previous and current disease states to transition string.

        Parameters
        ----------
        df
            The DataFrame containing the disease states.

        Returns
        -------
            The transitions between disease states.
        """
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
        """Update the previous state column to the current state.

        This enables tracking of transitions between states.
        """
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column_name] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    ###############
    # Aggregators #
    ###############

    def aggregate_state_person_time(self, x: pd.DataFrame) -> float:
        """Aggregate person time for the time step.

        Parameters
        ----------
        x
            The DataFrame containing the population.

        Returns
        -------
            The aggregated person time.
        """
        return len(x) * to_years(self.step_size())

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Rename the appropriate column to 'sub_entity'.

        The primary thing this method does is rename the appropriate column
        (either the transition stratification name of the disease name, depending
        on the measure) to 'sub_entity'. We do this here instead of the
        'get_sub_entity_column' method simply because we do not want the original
        column at all. If we keep it here and then return it as the sub-entity
        column later, the final results would have both.

        Parameters
        ----------
        measure
            The measure.
        results
            The results to format.

        Returns
        -------
            The formatted results.
        """
        results = results.reset_index()
        if "transition_count_" in measure:
            sub_entity = self.transition_stratification_name
        if "person_time_" in measure:
            sub_entity = self.disease
        results.rename(columns={sub_entity: COLUMNS.SUB_ENTITY}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'measure' column values."""
        if "transition_count_" in measure:
            measure_name = "transition_count"
        if "person_time_" in measure:
            measure_name = "person_time"
        return pd.Series(measure_name, index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity_type' column values."""
        return pd.Series(self.entity_type, index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity' column values."""
        return pd.Series(self.entity, index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'sub_entity' column values."""
        # The sub-entity col was created in the 'format' method
        return results[COLUMNS.SUB_ENTITY]
