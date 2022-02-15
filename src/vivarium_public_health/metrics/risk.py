"""
==============
Risk Observers
==============

This module contains tools for observing risk exposure during the simulation.

"""
from collections import Counter
from typing import Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

from vivarium_public_health.metrics.utilities import (
    get_age_bins,
    get_prevalent_cases,
    get_state_person_time,
)


class CategoricalRiskObserver:
    """An observer for a categorical risk factor.

    Observes category person time for a risk factor.

    By default, this observer computes aggregate categorical person time
    over the entire simulation.  It can be configured to bin these into
    age_groups, sexes, and years by setting the ``by_age``, ``by_sex``,
    and ``by_year`` flags, respectively.

    This component can also observe the number of simulants in each age
    group who are alive and in each category of risk at the specified
    sample date each year (the sample date defaults to July, 1, and can
    be set in the configuration).

    Here is an example configuration to change the sample date to Dec. 31:

    .. code-block:: yaml

        {risk_name}_observer:
            sample_date:
                month: 12
                day: 31
    """

    configuration_defaults = {
        "metrics": {
            "risk": {
                "by_age": False,
                "by_year": False,
                "by_sex": False,
                "sample_exposure": {
                    "sample": False,
                    "date": {
                        "month": 7,
                        "day": 1,
                    },
                },
            }
        }
    }

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk :
        the type and name of a risk, specified as "type.name". Type is singular.

        """
        self.risk = risk
        self.configuration_defaults = {
            "metrics": {
                f"{self.risk}": CategoricalRiskObserver.configuration_defaults["metrics"][
                    "risk"
                ]
            }
        }

    @property
    def name(self):
        return f"categorical_risk_observer.{self.risk}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.data = {}
        self.config = builder.configuration[f"metrics"][f"{self.risk}"]
        self.clock = builder.time.clock()
        self.categories = builder.data.load(f"risk_factor.{self.risk}.categories")
        self.age_bins = get_age_bins(builder)
        self.person_time = Counter()
        self.sampled_exposure = Counter()

        columns_required = ["alive"]
        if self.config.by_age:
            columns_required += ["age"]
        if self.config.by_sex:
            columns_required += ["sex"]
        self.population_view = builder.population.get_view(columns_required)

        self.exposure = builder.value.get_value(f"{self.risk}.exposure")
        builder.value.register_value_modifier("metrics", self.metrics)
        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)

    def on_time_step_prepare(self, event: Event):
        pop = pd.concat(
            [
                self.population_view.get(event.index),
                pd.Series(self.exposure(event.index), name=self.risk),
            ],
            axis=1,
        )

        for category in self.categories:
            state_person_time_this_step = get_state_person_time(
                pop,
                self.config,
                self.risk,
                category,
                self.clock().year,
                event.step_size,
                self.age_bins,
            )
            self.person_time.update(state_person_time_this_step)

        if self._should_sample(event.time):
            sampled_exposure = get_prevalent_cases(
                pop, self.config.to_dict(), self.risk, event.time, self.age_bins
            )
            self.sampled_exposure.update(sampled_exposure)

    def _should_sample(self, event_time: pd.Timestamp) -> bool:
        """Returns true if we should sample on this time step."""
        should_sample = self.config.sample_exposure.sample
        if should_sample:
            sample_date = pd.Timestamp(
                year=event_time.year, **self.config.sample_prevalence.date.to_dict()
            )
            should_sample &= self.clock() <= sample_date < event_time
        return should_sample

    # noinspection PyUnusedLocal
    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.person_time)
        metrics.update(self.sampled_exposure)
        return metrics

    def __repr__(self):
        return f"CategoricalRiskObserver({self.risk})"
