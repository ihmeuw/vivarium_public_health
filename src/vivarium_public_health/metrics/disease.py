"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""

from functools import partial
from typing import Any, Dict, List, Optional

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.results import StratifiedObserver

from vivarium_public_health.metrics.reporters import COLUMNS, write_dataframe_to_parquet
from vivarium_public_health.utilities import to_years


class DiseaseObserver(StratifiedObserver):
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
                self.disease: super().configuration_defaults["stratification"]["disease"]
            }
        }

    @property
    def columns_created(self) -> List[str]:
        return [self.previous_state_column_name]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [self.disease]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease
        self.current_state_column_name = self.disease
        self.previous_state_column_name = f"previous_{self.disease}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.disease]

    #################
    # Setup methods #
    #################

    def register_observations(self, builder):
        disease_model = builder.components.get_component(f"disease_model.{self.disease}")
        entity_type = disease_model.cause_type
        entity = disease_model.cause
        for state in disease_model.states:
            builder.results.register_observation(
                name=f"{state.state_id}_person_time",
                pop_filter=f'alive == "alive" and {self.disease} == "{state.state_id}" and tracked==True',
                aggregator=self.aggregate_state_person_time,
                requires_columns=["alive", self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
                report=partial(
                    self.write_disease_results,
                    measure_name="person_time",
                    entity_type=entity_type,
                    entity=entity,
                    sub_entity=state.state_id,
                ),
            )

        for transition in disease_model.transition_names:
            filter_string = (
                f'{self.previous_state_column_name} == "{transition.from_state}" '
                f'and {self.disease} == "{transition.to_state}" '
                f"and tracked==True "
                f'and alive == "alive"'
            )
            builder.results.register_observation(
                name=f"{transition}_event_count",
                pop_filter=filter_string,
                requires_columns=[self.previous_state_column_name, self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="collect_metrics",
                report=partial(
                    self.write_disease_results,
                    measure_name="transition_count",
                    entity_type=entity_type,
                    entity=entity,
                    sub_entity=transition,
                ),
            )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series("", index=pop_data.index, name=self.previous_state_column_name)
        )

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

    def write_disease_results(
        self,
        measure_name: str,
        entity_type: str,
        entity: str,
        sub_entity: str,
        measure: str,
        results: pd.DataFrame,
    ):
        """Combine each observation's results and save to a single file"""
        write_dataframe_to_parquet(
            results=results,
            measure=measure_name,
            entity_type=entity_type,
            entity=entity,
            sub_entity=sub_entity,
            results_dir=self.results_dir,
            random_seed=self.random_seed,
            input_draw=self.input_draw,
            output_filename=f"{measure_name}_{self.disease}",
        )
