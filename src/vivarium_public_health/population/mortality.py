"""
========================
The Core Mortality Model
========================

Summary
=======

The mortality component models all cause mortality and allows for disease
models to contribute cause specific mortality. At each timestep the currently
"alive" population is subjected to a mortality event using the mortality rate to
determine probabilities of death for each simulant. A weighted probable cause of
death is used to choose the cause of death. The years of life lost are
calculated by subtracting a simulant's age from the population TMRLE and the
population is updated.

Columns Created
===============

 - cause_of_death
 - years_of_life_lost

Pipelines Exposed
=================

 - cause_specific_mortality_rate
 - mortality_rate
 - affected_unmodeled.cause_specific_mortality_rate
 - affected_unmodeled.cause_specific_mortality_rate.paf


All cause mortality is read from the artifact (GBD). At setup cause specific
mortality is initialized to an empty table. As disease models are registered,
they affect cause specific mortality by means of
the cause_specific_mortality_rate pipeline. This is population level data.

If there are causes of death which are unmodeled, but may be impacted by some
modeled entity, they can be specified using the configuration key
"unmodeled_causes".

The mortality component's mortality_rate pipeline reflects the
cause deleted mortality rate (ACMR - CSMR). Then the impact of unmodeled causes
on mortality is calculated, by subtracting the raw unmodeled csmr before adding
back the modified unmodeled csmr.

"""

from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.utilities import get_lookup_columns


class Mortality(Component):
    """This is the mortality component which models of mortality in a population.

    The component models all cause mortality and allows for disease models to contribute
    cause specific mortality. Data used by this class should be supplied in the artifact
    and is configurable in the configuration to build lookup tables. For instance, let's
    say we want to use sex and hair color to build a lookup table for all cause mortality.

    .. code-block:: yaml

       configuration:
           mortality:
               all_cause_mortality_rate:
                   categorical_columns: ["sex", "hair_color"]

    Similarly, we can do the same thing for unmodeled causes. Here is an example:

    .. code-block:: yaml

       configuration:
           mortality:
               unmodeled_cause_specific_mortality_rate:
                   unmodeled_causes: ["maternal_disorders", maternal_hemorrhage]
                   categorical_columns: ["sex", "hair_color"]

    Or if we wanted to make the data a scalar value for all cause mortality rate we could
    configure that as well.

    .. code-block:: yaml

       configuration:
           mortality:
               all_cause_mortality_rate:
                   value: 0.01

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "mortality": {
                "data_sources": {
                    "all_cause_mortality_rate": "cause.all_causes.cause_specific_mortality_rate",
                    "unmodeled_cause_specific_mortality_rate": self.load_unmodeled_csmr,
                    "life_expectancy": "population.theoretical_minimum_risk_life_expectancy",
                },
                "unmodeled_causes": [],
            },
        }

    @property
    def standard_lookup_tables(self) -> list[str]:
        return [
            "all_cause_mortality_rate",
            "life_expectancy",
        ]

    @property
    def columns_created(self) -> list[str]:
        return [self.cause_of_death_column_name, self.years_of_life_lost_column_name]

    @property
    def columns_required(self) -> list[str] | None:
        return ["alive", "exit_time", "age", "sex"]

    @property
    def time_step_priority(self) -> int:
        return 0

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self._randomness_stream_name = "mortality_handler"

        self.cause_of_death_column_name = "cause_of_death"
        self.years_of_life_lost_column_name = "years_of_life_lost"

        self.cause_specific_mortality_rate_pipeline_name = "cause_specific_mortality_rate"
        self.mortality_rate_pipeline_name = "mortality_rate"
        self.unmodeled_csmr_pipeline_name = "affected_unmodeled.cause_specific_mortality_rate"
        self.unmodeled_csmr_paf_pipeline_name = f"{self.unmodeled_csmr_pipeline_name}.paf"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.random = self.get_randomness_stream(builder)
        self.clock = builder.time.clock()

        self.cause_specific_mortality_rate = self.get_cause_specific_mortality_rate(builder)

        self.unmodeled_csmr = self.get_unmodeled_csmr(builder)
        self.unmodeled_csmr_paf = self.get_unmodeled_csmr_paf(builder)
        self.mortality_rate = self.get_mortality_rate(builder)

    #################
    # Setup methods #
    #################

    def get_randomness_stream(self, builder) -> RandomnessStream:
        return builder.randomness.get_stream(self._randomness_stream_name)

    def get_cause_specific_mortality_rate(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.cause_specific_mortality_rate_pipeline_name,
            source=builder.lookup.build_table(0),
        )

    def get_mortality_rate(self, builder: Builder) -> Pipeline:
        required_columns = get_lookup_columns(
            [
                self.lookup_tables["all_cause_mortality_rate"],
                self.lookup_tables["unmodeled_cause_specific_mortality_rate"],
            ],
        )
        return builder.value.register_rate_producer(
            self.mortality_rate_pipeline_name,
            source=self.calculate_mortality_rate,
            requires_columns=required_columns,
        )

    def load_unmodeled_csmr(self, builder: Builder) -> float | pd.DataFrame:
        # todo validate that all data have the same columns
        raw_csmr = 0.0
        for idx, cause in enumerate(builder.configuration[self.name].unmodeled_causes):
            csmr = f"cause.{cause}.cause_specific_mortality_rate"
            if 0 == idx:
                raw_csmr = builder.data.load(csmr)
            else:
                raw_csmr.loc[:, "value"] += builder.data.load(csmr).value
        return raw_csmr

    def get_unmodeled_csmr(self, builder: Builder) -> Pipeline:
        required_columns = get_lookup_columns(
            [self.lookup_tables["unmodeled_cause_specific_mortality_rate"]]
        )
        return builder.value.register_value_producer(
            self.unmodeled_csmr_pipeline_name,
            source=self.get_unmodeled_csmr_source,
            requires_columns=required_columns,
        )

    def get_unmodeled_csmr_paf(self, builder: Builder) -> Pipeline:
        unmodeled_csmr_paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            self.unmodeled_csmr_paf_pipeline_name,
            source=lambda index: [unmodeled_csmr_paf(index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

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
        mortality_rates = self.mortality_rate(pop.index)
        mortality_hazard = mortality_rates.sum(axis=1)
        deaths = self.random.filter_for_rate(
            pop.index, mortality_hazard, additional_key="death"
        )
        if not deaths.empty:
            cause_of_death_weights = mortality_rates.loc[deaths, :].divide(
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
            pop.loc[deaths, "years_of_life_lost"] = self.lookup_tables["life_expectancy"](
                deaths
            )
            pop.loc[deaths, "cause_of_death"] = cause_of_death
            self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def calculate_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        acmr = self.lookup_tables["all_cause_mortality_rate"](index)
        modeled_csmr = self.cause_specific_mortality_rate(index)
        unmodeled_csmr_raw = self.lookup_tables["unmodeled_cause_specific_mortality_rate"](
            index
        )
        unmodeled_csmr = self.unmodeled_csmr(index)
        cause_deleted_mortality_rate = (
            acmr - modeled_csmr - unmodeled_csmr_raw + unmodeled_csmr
        )
        return pd.DataFrame({"other_causes": cause_deleted_mortality_rate})

    def get_unmodeled_csmr_source(self, index: pd.Index) -> pd.Series:
        raw_csmr = self.lookup_tables["unmodeled_cause_specific_mortality_rate"](index)
        paf = self.unmodeled_csmr_paf(index)
        return raw_csmr * (1 - paf)
