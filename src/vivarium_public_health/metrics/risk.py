"""
==============
Risk Observers
==============

This module contains tools for observing risk exposure during the simulation.

"""
from collections import Counter
from typing import Dict, List

import pandas as pd
from vivarium.config_tree import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.values import Pipeline

from vivarium_public_health.metrics.stratification import ResultsStratifier
from vivarium_public_health.utilities import to_years


class CategoricalRiskObserver:
    """An observer for a categorical risk factor.

    Observes category person time for a risk factor.

    By default, this observer computes aggregate categorical person time
    over the full course of the simulation. It can be configured to add or
    remove stratification groups to the default groups defined by a
    ResultsStratifier.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            stratification:
                risk_name:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"
    """

    configuration_defaults = {
        "stratification": {
            "risk": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
        name of a risk

        """
        self.risk = risk
        self.configuration_defaults = self._get_configuration_defaults()

        self.exposure_pipeline_name = f"{self.risk}.exposure"

    def __repr__(self):
        return f"CategoricalRiskObserver({self.risk})"

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "stratification": {
                f"{self.risk}": CategoricalRiskObserver.configuration_defaults[
                    "stratification"
                ]["risk"]
            }
        }

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return f"categorical_risk_observer.{self.risk}"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.risk]
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")

        columns_required = ["alive"]
        self.population_view = builder.population.get_view(columns_required)

        for category in self.categories:
            builder.results.register_observation(
                name=f"{self.risk}_{category}_person_time",
                pop_filter=f'alive == "alive" and `{self.exposure_pipeline_name}`=="{category}" and tracked==True',
                aggregator=self.aggregate_risk_category_person_time,
                requires_columns=["alive"],
                requires_values=[self.exposure_pipeline_name],
                additional_stratifications=self.config.include,
                excluded_stratifications=self.config.exclude,
                when="time_step__prepare",
            )

    def aggregate_risk_category_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())

    def _get_stratification_configuration(self, builder: Builder) -> ConfigTree:
        return builder.configuration.observers[self.risk]
