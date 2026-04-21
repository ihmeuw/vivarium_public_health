"""
========================
The Core Mortality Model
========================

The mortality component models :term:`all-cause mortality <ACMR>` and allows
for disease models to contribute :term:`cause-specific mortality <CSMR>`. At
each time step, the currently-alive population is subjected to a mortality event
using the :term:`mortality rate <Mortality Rate>` to determine probabilities of
death for each simulant.

A weighted probable cause of death is used to choose the cause of death. The
:term:`years of life lost <YLL>` are calculated by subtracting a
simulant's age from the population :term:`TMRLE` and the population is updated.

:term:`ACMR` is read from the artifact (GBD). At setup :term:`cause-specific
mortality <CSMR>` is initialized to an empty table. As disease models are
registered, they affect :term:`CSMR` by means of the
cause_specific_mortality_rate pipeline. This is population level data.

If there are causes of death which are :term:`unmodeled <Unmodeled Cause>`, but
may be impacted by some modeled entity, they can be specified using the
configuration key "unmodeled_causes".

The mortality component's mortality_rate pipeline reflects the
:term:`cause-deleted <Cause-Deleted Mortality>` mortality rate (ACMR - CSMR).
Then the impact of :term:`unmodeled causes <Unmodeled Cause>` on mortality is
calculated, by subtracting the raw unmodeled CSMR before adding back the
modified unmodeled CSMR.

"""

from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream

from vivarium_public_health.causal_factor.calibration_constant import (
    register_risk_affected_attribute_producer,
)


class Mortality(Component):
    """Model mortality in a population.

    This component models :term:`all-cause mortality <ACMR>` and allows for disease
    models to contribute :term:`cause-specific mortality <CSMR>`. Data used by this
    class should be supplied in the artifact and is configurable in the configuration
    to build lookup tables. For instance, let's say we want to use sex and hair color
    to build a lookup table for all cause mortality.

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
                data_sources:
                    all_cause_mortality_rate: 0.01

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """The default configuration values for this component.

        Configuration structure::

            mortality:
                data_sources:
                    all_cause_mortality_rate:
                        Source for all-cause mortality rate data. Default is
                        the artifact key
                        ``cause.all_causes.cause_specific_mortality_rate``.
                        This represents the background mortality rate from
                        all causes combined.
                    unmodeled_cause_specific_mortality_rate:
                        Source for unmodeled CSMR data. Default uses the
                        ``load_unmodeled_csmr`` method which sums CSMRs for
                        all causes listed in ``unmodeled_causes``.
                    life_expectancy:
                        Source for life expectancy data. Default is the
                        artifact key
                        ``population.theoretical_minimum_risk_life_expectancy``.
                        Used to calculate :term:`years of life lost <Years of Life Lost>` (YLLs).
                unmodeled_causes: list[str]
                    List of cause names that are not explicitly modeled but
                    may be affected by modeled risks. Their CSMRs are
                    combined into a single pipeline that can be modified.
                    Default is an empty list.
        """
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
        """The names of lookup tables built automatically by the framework."""
        return [
            "all_cause_mortality_rate",
            "life_expectancy",
        ]

    @property
    def time_step_priority(self) -> int:
        """The time step priority for mortality processing.

        It is set to 6 so that observations (priority 5) record person-time
        while simulants are still alive, before mortality removes them.
        """
        return 6

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self._randomness_stream_name = "mortality_handler"

        self.cause_of_death_column_name = "cause_of_death"
        self.years_of_life_lost_column_name = "years_of_life_lost"

        self.cause_specific_mortality_rate_pipeline = "cause_specific_mortality_rate"
        self.mortality_rate_pipeline = "mortality_rate"
        self.unmodeled_csmr_pipeline = "affected_unmodeled.cause_specific_mortality_rate"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Set up the component by registering pipelines, lookup tables, and the population initializer.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.random = self.get_randomness_stream(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        self.acmr_table = self.build_lookup_table(builder, "all_cause_mortality_rate")
        self.unmodeled_csmr_table = self.build_lookup_table(
            builder, "unmodeled_cause_specific_mortality_rate"
        )
        self.life_expectancy_table = self.build_lookup_table(builder, "life_expectancy")

        self.register_cause_specific_mortality_rate(builder)

        self.register_unmodeled_csmr(builder)
        self.register_mortality_rate(builder)

        builder.value.register_attribute_modifier("exit_time", self.update_exit_times)

        builder.population.register_initializer(
            initializer=self.initialize_mortality,
            columns=[
                "is_alive",
                self.cause_of_death_column_name,
                self.years_of_life_lost_column_name,
            ],
        )

    #################
    # Setup methods #
    #################

    def get_randomness_stream(self, builder: Builder) -> RandomnessStream:
        """Get the randomness stream used for stochastic mortality events.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The :class:`~vivarium.framework.randomness.stream.RandomnessStream` used
            for filtering deaths and choosing cause of death.
        """
        return builder.randomness.get_stream(self._randomness_stream_name)

    def register_cause_specific_mortality_rate(self, builder: Builder) -> None:
        """Register the cause-specific mortality rate attribute pipeline.

        Creates an attribute producer for the ``cause_specific_mortality_rate``
        pipeline initialized to zero. Disease models may register modifiers on
        this pipeline to contribute their own cause-specific rates.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self.cause_specific_mortality_rate_pipeline,
            source=self.build_lookup_table(builder, "csmr", 0),
        )

    def register_mortality_rate(self, builder: Builder) -> None:
        """Register the :term:`mortality rate <Mortality Rate>` attribute pipeline.

        The attribute pipeline source is :meth:`calculate_mortality_rate`, which computes
        the mortality rate from the all-cause rate and the registered cause-specific rates.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_rate_producer(
            self.mortality_rate_pipeline,
            source=self.calculate_mortality_rate,
            required_resources=[self.acmr_table, self.unmodeled_csmr_table],
        )

    def load_unmodeled_csmr(self, builder: Builder) -> float | pd.DataFrame:
        """Load and sum the cause-specific mortality rates for all unmodeled causes.

        Iterates over causes listed in ``configuration.mortality.unmodeled_causes``
        and accumulates their CSMRs from the artifact. Returns ``0.0`` if no unmodeled
        causes are configured.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The summed CSMR for all unmodeled causes as a :class:`pandas.DataFrame`,
            or ``0.0`` if there are no unmodeled causes.
        """
        # todo validate that all data have the same columns
        raw_csmr = 0.0
        for idx, cause in enumerate(builder.configuration[self.name].unmodeled_causes):
            csmr = f"cause.{cause}.cause_specific_mortality_rate"
            if 0 == idx:
                raw_csmr = builder.data.load(csmr)
            else:
                raw_csmr.loc[:, "value"] += builder.data.load(csmr).value
        return raw_csmr

    def register_unmodeled_csmr(self, builder: Builder) -> None:
        """Register the unmodeled cause-specific mortality rate pipeline.

        Creates a risk-affected attribute pipeline for the
        :term:`unmodeled <Unmodeled Cause>` :term:`CSMR` pipeline so that
        modeled risks can apply modifiers to unmodeled causes.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        register_risk_affected_attribute_producer(
            builder=builder,
            name=self.unmodeled_csmr_pipeline,
            source=self.unmodeled_csmr_table,
        )

    def update_exit_times(self, index: pd.Index, previous_exit_time: pd.Series) -> pd.Series:
        """Update exit times for simulants who have died."""
        dead_idx = self.population_view.get_filtered_index(index, query="is_alive == False")
        newly_dead_idx = dead_idx.intersection(
            previous_exit_time[previous_exit_time.isna()].index
        )
        previous_exit_time.loc[newly_dead_idx] = self.clock() + self.step_size()
        return previous_exit_time

    ########################
    # Event-driven methods #
    ########################

    def initialize_mortality(self, pop_data: SimulantData) -> None:
        """Initialize mortality-related columns for new simulants.

        Parameters
        ----------
        pop_data
            Metadata about the simulants being initialized.
        """
        pop_update = pd.DataFrame(
            {
                "is_alive": True,
                self.cause_of_death_column_name: "not_dead",
                self.years_of_life_lost_column_name: 0.0,
            },
            index=pop_data.index,
        )
        self.population_view.initialize(pop_update)

    def on_time_step(self, event: Event) -> None:
        """Apply mortality to the living population at each time step.

        Determines which simulants die based on the current :term:`mortality rate <Mortality Rate>`,
        then probabilistically assigns a cause of death and calculates
        :term:`years of life lost <Years of Life Lost>` for the deceased simulants.

        Parameters
        ----------
        event
            The event that triggered this method call.
        """
        living_idx = self.population_view.get_filtered_index(
            event.index, query="is_alive == True"
        )
        mortality_rates = self.population_view.get_frame(
            living_idx, self.mortality_rate_pipeline
        )
        mortality_hazard = mortality_rates.sum(axis=1)
        deaths = self.random.filter_for_rate(
            living_idx, mortality_hazard, additional_key="death"
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
            life_expectancy = self.life_expectancy_table(deaths)

            def _apply_deaths(pop: pd.DataFrame) -> pd.DataFrame:
                dead = pop.loc[deaths]
                dead["is_alive"] = False
                dead[self.years_of_life_lost_column_name] = life_expectancy
                dead[self.cause_of_death_column_name] = cause_of_death
                return dead

            self.population_view.update(
                [
                    "is_alive",
                    self.years_of_life_lost_column_name,
                    self.cause_of_death_column_name,
                ],
                _apply_deaths,
            )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def calculate_mortality_rate(self, index: pd.Index) -> pd.DataFrame:
        """Compute the :term:`mortality rate <Mortality Rate>` for the given simulants.

        The mortality rate is calculated as a :term:`cause-deleted mortality rate <Cause-Deleted Mortality>`:

        ``ACMR - modeled_CSMR - unmodeled_CSMR_raw + unmodeled_CSMR_modified``

        where ``unmodeled_CSMR_modified`` is the attribute calculated after any
        risk modifiers have been applied.

        Parameters
        ----------
        index
            Index of the simulants for whom to compute the rate.

        Returns
        -------
            A :class:`pandas.DataFrame` with a single column ``'other_causes'``
            containing the mortality rate for each simulant.
        """
        acmr = self.acmr_table(index)
        modeled_csmr = self.population_view.get(
            index, self.cause_specific_mortality_rate_pipeline
        )
        unmodeled_csmr_raw = self.unmodeled_csmr_table(index)
        unmodeled_csmr = self.population_view.get(index, self.unmodeled_csmr_pipeline)
        cause_deleted_mortality_rate = (
            acmr - modeled_csmr - unmodeled_csmr_raw + unmodeled_csmr
        )
        return pd.DataFrame({"other_causes": cause_deleted_mortality_rate})
