"""
========================
The Core Mortality Model
========================

This module contains tools modeling all cause mortality and hooks for
disease models to contribute cause-specific and excess mortality.

"""
from typing import Callable, Dict, List, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.time import Time
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor


class Mortality:

    configuration_defaults = {"unmodeled_causes": []}

    def __init__(self):
        self._randomness_stream_name = "mortality_handler"
        self.cause_specific_mortality_rate_pipeline_name = "cause_specific_mortality_rate"
        self.mortality_rate_pipeline_name = "mortality_rate"
        self.cause_of_death_column_name = "cause_of_death"
        self.years_of_life_lost_column_name = "years_of_life_lost"
        self.unmodeled_csmr_pipeline_name = "affected_unmodeled.cause_specific_mortality_rate"
        self.unmodeled_csmr_paf_pipeline_name = f"{self.unmodeled_csmr_pipeline_name}.paf"
        self.all_cause_mortality_hazard_pipeline_name = "all_causes.mortality_hazard"
        self.all_cause_mortality_hazard_paf_pipeline_name = (
            f"{self.all_cause_mortality_hazard_pipeline_name}.paf"
        )

    def __repr__(self) -> str:
        return f"Mortality()"

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, List]:
        return {"unmodeled_causes": Mortality.configuration_defaults["unmodeled_causes"]}

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

        self.cause_specific_mortality_rate = self._get_cause_specific_mortality_rate(builder)
        self.mortality_rate = self._get_mortality_rate(builder)

        self.all_cause_mortality_rate = self._get_all_cause_mortality_rate(builder)
        self.life_expectancy = self._get_life_expectancy(builder)

        self._raw_unmodeled_csmr = self._get_raw_unmodeled_csmr(builder)
        self.unmodeled_csmr = self._get_unmodeled_csmr(builder)
        self.unmodeled_csmr_paf = self._get_unmodeled_csmr_paf(builder)
        self.mortality_hazard = self._get_mortality_hazard(builder)
        self._mortality_hazard_paf = self._get_mortality_hazard_paf(builder)

        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)
        self._register_on_timestep_listener(builder)

    def _get_randomness_stream(self, builder) -> RandomnessStream:
        return builder.randomness.get_stream(self._randomness_stream_name)

    # noinspection PyMethodMayBeStatic
    def _get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

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
    def _get_all_cause_mortality_rate(self, builder: Builder) -> Union[LookupTable, Pipeline]:
        acmr_data = builder.data.load("cause.all_causes.cause_specific_mortality_rate")
        return builder.lookup.build_table(
            acmr_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    # noinspection PyMethodMayBeStatic
    def _get_life_expectancy(self, builder: Builder) -> Union[LookupTable, Pipeline]:
        life_expectancy_data = builder.data.load(
            "population.theoretical_minimum_risk_life_expectancy"
        )
        return builder.lookup.build_table(life_expectancy_data, parameter_columns=["age"])

    # noinspection PyMethodMayBeStatic
    def _get_raw_unmodeled_csmr(self, builder: Builder) -> Union[LookupTable, Pipeline]:
        unmodeled_causes = builder.configuration.unmodeled_causes
        raw_csmr = 0.0
        for idx, cause in enumerate(unmodeled_causes):
            csmr = f"cause.{cause}.cause_specific_mortality_rate"
            if 0 == idx:
                raw_csmr = builder.data.load(csmr)
            else:
                raw_csmr.loc[:, "value"] += builder.data.load(csmr).value

        additional_parameters = (
            {"key_columns": ["sex"], "parameter_columns": ["age", "year"]}
            if unmodeled_causes
            else {}
        )

        return builder.lookup.build_table(raw_csmr, **additional_parameters)

    def _get_unmodeled_csmr(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.unmodeled_csmr_pipeline_name,
            source=self._get_unmodeled_csmr_source,
            requires_columns=["age", "sex"],
        )

    def _get_unmodeled_csmr_paf(self, builder: Builder) -> Pipeline:
        unmodeled_csmr_paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            self.unmodeled_csmr_paf_pipeline_name,
            source=lambda index: [unmodeled_csmr_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def _get_mortality_hazard(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.all_cause_mortality_hazard_pipeline_name,
            source=self._get_mortality_hazard_source,
        )

    def _get_mortality_hazard_paf(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.all_cause_mortality_hazard_paf_pipeline_name,
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(
            [
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
                "alive",
                "exit_time",
                "age",
                "sex",
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
        mortality_hazard = self.mortality_hazard(pop.index)
        deaths = self.random.filter_for_rate(
            pop.index, mortality_hazard, additional_key="death"
        )
        if not deaths.empty:
            cause_of_death_weights = self.mortality_rate(deaths).divide(
                mortality_hazard.loc[deaths], axis=0
            )
            cause_of_death = self.random.choice(
                deaths,
                cause_of_death_weights.columns,
                cause_of_death_weights,
                additional_key="cause_of_death",
            )
            pop.loc[deaths, "alive"] = "dead"
            pop.loc[deaths, "exit_time"] = event.time
            pop.loc[deaths, "years_of_life_lost"] = self.life_expectancy(deaths)
            pop.loc[deaths, "cause_of_death"] = cause_of_death
            self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _calculate_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        acmr = self.all_cause_mortality_rate(index)
        modeled_csmr = self.cause_specific_mortality_rate(index)
        unmodeled_csmr_raw = self._raw_unmodeled_csmr(index)
        unmodeled_csmr = self.unmodeled_csmr(index)
        cause_deleted_mortality_rate = (
            acmr - modeled_csmr - unmodeled_csmr_raw + unmodeled_csmr
        )
        return pd.DataFrame({"other_causes": cause_deleted_mortality_rate})

    def _get_unmodeled_csmr_source(self, index: pd.Index) -> pd.Series:
        raw_csmr = self._raw_unmodeled_csmr(index)
        paf = self.unmodeled_csmr_paf(index)
        return raw_csmr * (1 - paf)

    def _get_mortality_hazard_source(self, index: pd.Index) -> pd.Series:
        mortality_rates = pd.DataFrame(self.mortality_rate(index))
        mortality_hazard = mortality_rates.sum(axis=1)
        paf = self._mortality_hazard_paf(index)
        return mortality_hazard * (1 - paf)
