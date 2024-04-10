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
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.utilities import get_lookup_columns


class Mortality(Component):

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "mortality": {
                "lookup_tables": {
                    "all_cause_mortality_rate": self.build_lookup_table_config(
                        value="data",
                        continuous_columns=["age", "year"],
                        categorical_columns=["sex"],
                        key_name="cause.all_causes.all_cause_mortality_rate",
                    ),
                    "unmodeled_cause_specific_mortality_rate": self.build_lookup_table_config(
                        value="data",
                        continuous_columns=["age", "year"],
                        categorical_columns=["sex"],
                        skip_build=True,
                        **{"unmodeled_causes": []},
                    ),
                    "life_expectancy": self.build_lookup_table_config(
                        value="data",
                        continuous_columns=["age"],
                        categorical_columns=[],
                        key_name="population.theoretical_minimum_risk_life_expectancy",
                    ),
                },
            },
        }

    @property
    def columns_created(self) -> List[str]:
        return [self.cause_of_death_column_name, self.years_of_life_lost_column_name]

    @property
    def columns_required(self) -> Optional[List[str]]:
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

    def create_lookup_tables(self, builder: "Builder") -> None:
        """
        Create lookup tables for the mortality component.

        Parameters
        ----------
        builder
            Interface to access simulation managers.
        """
        super().create_lookup_tables(builder)
        self.unmodeled_cause_specific_mortality_rate = self.get_raw_unmodeled_csmr(builder)
        self.lookup_tables[
            "unmodeled_cause_specific_mortality_rate"
        ] = self.unmodeled_cause_specific_mortality_rate

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

    # noinspection PyMethodMayBeStatic
    def get_raw_unmodeled_csmr(self, builder: Builder) -> Union[LookupTable, Pipeline]:
        """
        Load unmodeled cause specific mortality rate data and build a lookup
        table or pipeline.

        Parameters
        ----------
        builder
            Interface to access simulation managers.

        Returns
        -------
        Union[LookupTable, Pipeline]
            A lookup table or pipeline returning the unmodeled csmr.
        """
        unmodeled_causes_config = (
            builder.configuration.mortality.lookup_tables.unmodeled_cause_specific_mortality_rate
        )
        raw_csmr = 0.0
        for idx, cause in enumerate(unmodeled_causes_config.unmodeled_causes):
            csmr = f"cause.{cause}.cause_specific_mortality_rate"
            if 0 == idx:
                raw_csmr = builder.data.load(csmr)
            else:
                raw_csmr.loc[:, "value"] += builder.data.load(csmr).value

        additional_parameters = (
            {
                "key_columns": unmodeled_causes_config["categorical_columns"],
                "parameter_columns": unmodeled_causes_config["continuous_columns"],
            }
            if unmodeled_causes_config["unmodeled_causes"]
            else {}
        )

        return builder.lookup.build_table(raw_csmr, **additional_parameters)

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
