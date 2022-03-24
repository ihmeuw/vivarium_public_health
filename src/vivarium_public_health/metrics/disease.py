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
from vivarium.framework.event import Event

from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.metrics.utilities import (
    TransitionString,
    get_age_bins,
    get_state_person_time,
    get_transition_count,
)


class DiseaseObserver:
    """Observes disease counts, person time, and prevalent cases for a cause.

    By default, this observer computes aggregate susceptible person time
    and counts of disease cases over the entire simulation.  It can be
    configured to bin these into age_groups, sexes, and years by setting
    the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    It also can record prevalent cases on a particular sample date each year,
    though by default this is disabled. These will also be binned based on the
    flags set for the observer. Additionally, the sample date is configurable
    and defaults to July 1st of each year.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                {YOUR_DISEASE_NAME}_observer:
                    by_age: True
                    by_year: False
                    by_sex: True
                    sample_prevalence:
                        sample: True
                        date:
                            month: 4
                            day: 10

    """

    configuration_defaults = {
        "metrics": {
            "disease_observer": {
                "by_age": False,
                "by_year": False,
                "by_sex": False,
                "sample_prevalence": {
                    "sample": False,
                    "date": {
                        "month": 7,
                        "day": 1,
                    },
                },
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.configuration_defaults = {
            "metrics": {
                f"{disease}_observer": DiseaseObserver.configuration_defaults["metrics"][
                    "disease_observer"
                ]
            }
        }
        self.stratifier = ResultsStratifier(self.name)

    @property
    def name(self):
        return f"disease_observer.{self.disease}"

    @property
    def sub_components(self) -> List:
        return [self.stratifier]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration["metrics"][f"{self.disease}_observer"]
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()
        self.prevalence = Counter()

        comp = builder.components.get_component(f"disease_model.{self.disease}")
        self.states = comp.state_names
        self.transitions = comp.transition_names

        self.previous_state_column = f"previous_{self.disease}"
        builder.population.initializes_simulants(
            self.on_initialize_simulants, creates_columns=[self.previous_state_column]
        )

        columns_required = ["alive", f"{self.disease}", self.previous_state_column]
        if self.config.by_age:
            columns_required += ["age"]
        if self.config.by_sex:
            columns_required += ["sex"]
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier("metrics", self.metrics)
        # FIXME: The state table is modified before the clock advances.
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.event.register_listener("collect_metrics", self.on_collect_metrics)

    def on_initialize_simulants(self, pop_data: pd.DataFrame) -> None:
        self.population_view.update(
            pd.Series("", index=pop_data.index, name=self.previous_state_column)
        )

    def on_time_step_prepare(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for labels, pop_in_group in self.stratifier.group(pop):
            for state in self.states:
                # noinspection PyTypeChecker
                state_person_time_this_step = get_state_person_time(
                    pop_in_group,
                    self.config,
                    self.disease,
                    state,
                    self.clock().year,
                    event.step_size,
                    self.age_bins,
                )
                state_person_time_this_step = self.stratifier.update_labels(
                    state_person_time_this_step, labels
                )
                self.person_time.update(state_person_time_this_step)

        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        for labels, pop_in_group in self.stratifier.group(pop):
            for transition in self.transitions:
                transition = TransitionString(transition)
                # noinspection PyTypeChecker
                transition_counts_this_step = get_transition_count(
                    pop_in_group,
                    self.config,
                    self.disease,
                    transition,
                    event.time,
                    self.age_bins,
                )
                transition_counts_this_step = self.stratifier.update_labels(
                    transition_counts_this_step, labels
                )
                self.counts.update(transition_counts_this_step)

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.counts)
        metrics.update(self.person_time)
        metrics.update(self.prevalence)
        return metrics

    def __repr__(self):
        return "DiseaseObserver()"
