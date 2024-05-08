"""
================
Disease Observer
================

This module contains tools for observing disease incidence and prevalence
in the simulation.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.results import METRICS_COLUMN, StratifiedObserver
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
        for state in disease_model.states:
            builder.results.register_observation(
                name=f"{state.state_id}_person_time",
                pop_filter=f'alive == "alive" and {self.disease} == "{state.state_id}" and tracked==True',
                aggregator=self.aggregate_state_person_time,
                requires_columns=["alive", self.disease],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
                report=self.report_person_time,
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
                report=self.report_event_count,
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

    def report_person_time(self, measure: str, results: pd.DataFrame):
        results_dir = Path(self.results_dir)
        state = measure.split("_person_time")[0]
        measure = "state_person_time"
        # Add extra cols
        col_map = {
            "measure": measure,
            "state": state,
            "random_seed": self.random_seed,
            "input_draw": self.input_draw,
        }
        for col, val in col_map.items():
            if val is not None:
                results[col] = val
        # Sort the columns such that the stratifications (index) are first
        # and METRICS_COLUMN is last and sort the rows by the stratifications.
        other_cols = [c for c in results.columns if c != METRICS_COLUMN]
        results = results[other_cols + [METRICS_COLUMN]].sort_index().reset_index()

        # Concat and save
        results_file = results_dir / f"{measure}.csv"
        if not results_file.exists():
            results.to_csv(results_file, index=False)
        else:
            results.to_csv(
                results_dir / f"{measure}.csv", index=False, mode="a", header=False
            )

    def report_event_count(self, measure: str, results: pd.DataFrame):
        results_dir = Path(self.results_dir)
        transition = measure.split("_event_count")[0]
        measure = "transition_count"
        # Add extra cols
        col_map = {
            "measure": measure,
            "transition": transition,
            "random_seed": self.random_seed,
            "input_draw": self.input_draw,
        }
        for col, val in col_map.items():
            if val is not None:
                results[col] = val
        # Sort the columns such that the stratifications (index) are first
        # and METRICS_COLUMN is last and sort the rows by the stratifications.
        other_cols = [c for c in results.columns if c != METRICS_COLUMN]
        results = results[other_cols + [METRICS_COLUMN]].sort_index().reset_index()

        # Concat and save
        results_file = results_dir / f"{measure}.csv"
        if not results_file.exists():
            results.to_csv(results_file, index=False)
        else:
            results.to_csv(
                results_dir / f"{measure}.csv", index=False, mode="a", header=False
            )
