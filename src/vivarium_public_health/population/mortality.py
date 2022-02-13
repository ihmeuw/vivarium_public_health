"""
========================
The Core Mortality Model
========================

This module contains tools modeling all cause mortality and hooks for
disease models to contribute cause-specific and excess mortality.

"""
from typing import Callable

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.time import Time
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import Pipeline


class Mortality:
    def __init__(self):
        self._randomness_stream_name = "mortality_handler"
        self.cause_specific_mortality_rate_pipeline_name = "cause_specific_mortality_rate"
        self.mortality_rate_pipeline_name = "mortality_rate"
        self.cause_of_death_column_name = "cause_of_death"
        self.years_of_life_lost_column_name = "years_of_life_lost"

    def __repr__(self) -> str:
        return f"Mortality()"

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return "mortality"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.random = self._get_randomness_stream(builder)
        self.clock = self._get_clock(builder)
        self.all_cause_mortality_rate = self._get_all_cause_mortality_rate(builder)
        self.cause_specific_mortality_rate = self._get_cause_specific_mortality_rate(builder)
        self.mortality_rate = self._get_mortality_rate(builder)
        self.life_expectancy = self._get_life_expectancy(builder)
        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)
        self._register_on_timestep_listener(builder)

    def _get_randomness_stream(self, builder) -> RandomnessStream:
        return builder.randomness.get_stream(self._randomness_stream_name)

    # noinspection PyMethodMayBeStatic
    def _get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def _get_all_cause_mortality_rate(self, builder: Builder) -> LookupTable:
        acmr_data = builder.data.load("cause.all_causes.cause_specific_mortality_rate")
        return builder.lookup.build_table(
            acmr_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def _get_cause_specific_mortality_rate(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.cause_specific_mortality_rate_pipeline_name,
            source=builder.lookup.build_table(0),
        )

    def _get_mortality_rate(self, builder: Builder) -> Pipeline:
        return builder.value.register_rate_producer(
            self.mortality_rate_pipeline_name,
            source=self._calculate_mortality_rate,
            requires_columns=["age", "sex"],
        )

    # noinspection PyMethodMayBeStatic
    def _get_life_expectancy(self, builder: Builder) -> LookupTable:
        life_expectancy_data = builder.data.load(
            "population.theoretical_minimum_risk_life_expectancy"
        )
        return builder.lookup.build_table(life_expectancy_data, parameter_columns=["age"])

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
                "alive",
                "exit_time",
                "age",
                "sex",
                "location",
            ]
        )

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
            ],
        )

    def _register_on_timestep_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_time_step, priority=0)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame(
            {
                self.cause_of_death_column_name: "not_dead",
                self.years_of_life_lost_column_name: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index, query="alive =='alive'")
        prob_df = rate_to_probability(pd.DataFrame(self.mortality_rate(pop.index)))
        prob_df["no_death"] = 1 - prob_df.sum(axis=1)
        prob_df["cause_of_death"] = self.random.choice(
            prob_df.index, prob_df.columns, prob_df
        )
        dead_pop = prob_df.query('cause_of_death != "no_death"').copy()

        if not dead_pop.empty:
            dead_pop["alive"] = pd.Series("dead", index=dead_pop.index)
            dead_pop["exit_time"] = event.time
            dead_pop["years_of_life_lost"] = self.life_expectancy(dead_pop.index)
            self.population_view.update(
                dead_pop[["alive", "exit_time", "cause_of_death", "years_of_life_lost"]]
            )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _calculate_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        acmr = self.all_cause_mortality_rate(index)
        csmr = self.cause_specific_mortality_rate(index, skip_post_processor=True)
        cause_deleted_mortality_rate = acmr - csmr
        return pd.DataFrame({"other_causes": cause_deleted_mortality_rate})
