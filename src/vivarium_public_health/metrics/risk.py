"""
==============
Risk Observers
==============

This module contains tools for observing risk exposure during the simulation.

"""
from typing import Any, Dict, List, Optional

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder

from vivarium_public_health.utilities import to_years


class CategoricalRiskObserver(Component):
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

    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "risk": {
                "exclude": [],
                "include": [],
            }
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return {
            "stratification": {
                f"{self.risk}": self.CONFIGURATION_DEFAULTS["stratification"]["risk"]
            }
        }

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["alive"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
        name of a risk

        """
        super().__init__()
        self.risk = risk
        self.exposure_pipeline_name = f"{self.risk}.exposure"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.step_size = builder.time.step_size()
        self.config = builder.configuration.stratification[self.risk]
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")

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

    ###############
    # Aggregators #
    ###############

    def aggregate_risk_category_person_time(self, x: pd.DataFrame) -> float:
        return len(x) * to_years(self.step_size())
