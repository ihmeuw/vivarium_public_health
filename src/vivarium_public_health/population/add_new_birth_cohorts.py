"""
================
Fertility Models
================

Provide several different models of fertility.

"""

from typing import Any

import numpy as np
import pandas as pd
from vivarium.config_tree import ConfigTree
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.event import Event
from vivarium.engine.framework.population import SimulantData

from vivarium_public_health import utilities

# TODO: Incorporate better data into gestational model (probably as a separate component)
PREGNANCY_DURATION = pd.Timedelta(days=9 * utilities.DAYS_PER_MONTH)


class FertilityDeterministic(Component):
    """Deterministic model of births based on a fixed yearly simulant count.

    At each time step this component adds a fixed number of newborn simulants
    proportional to ``fertility.number_of_new_simulants_each_year`` and the
    current step size. Sub-integer remainders are accumulated across steps
    and applied when they reach a whole number.

    """

    CONFIGURATION_DEFAULTS = {
        "fertility": {
            "number_of_new_simulants_each_year": 1000,
        },
    }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        """Set up the component by reading configuration and obtaining the simulant creator.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.fractional_new_births = 0
        self.simulants_per_year = (
            builder.configuration.fertility.number_of_new_simulants_each_year
        )

        self.simulant_creator = builder.population.get_simulant_creator()

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """Add a set number of simulants to the population each time step.

        Parameters
        ----------
        event
            The event that triggered this method call.
        """
        # Assume births are uniformly distributed throughout the year.
        step_size = utilities.to_years(event.step_size)
        simulants_to_add = self.simulants_per_year * step_size + self.fractional_new_births

        self.fractional_new_births = simulants_to_add % 1
        simulants_to_add = int(simulants_to_add)

        if simulants_to_add > 0:
            self.simulant_creator(
                simulants_to_add,
                {
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                },
            )


class FertilityCrudeBirthRate(Component):
    """Population-level model of births using :term:`crude birth rate <Crude Birth Rate>`.

    The number of births added each time step is calculated as:

        new_births = sim_pop_size_t0 * live_births / true_pop_size * step_size

    Where:

    - ``sim_pop_size_t0`` - the initial simulation population size
    - ``live_births`` - annual number of live births in the true population
    - ``true_pop_size`` - the true population size

    This component has configuration flags that determine whether the
    live births and the true population size should vary with time.

    Notes
    -----
    The OECD definition of crude birth rate can be found on their
    `website <https://stats.oecd.org/glossary/detail.asp?ID=490>`_, while a more
    thorough discussion of fertility and birth rate models can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Birth_rate>`_ or in demography textbooks.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """The default configuration values for this component.

        Configuration structure::

            fertility:
                data_sources:
                    population_structure:
                        Source for population structure data. Default is the
                        ``population.structure`` artifact key.
                    live_births_by_sex:
                        Source for live births data. Default is the
                        ``covariate.live_births_by_sex.estimate`` artifact key.
                time_dependent_live_births: bool
                    Whether live births should vary with time. Default True.
                time_dependent_population_fraction: bool
                    Whether population fractions should vary with time.
                    Default False.
        """
        return {
            "fertility": {
                "data_sources": {
                    "population_structure": "population.structure",
                    "live_births_by_sex": "covariate.live_births_by_sex.estimate",
                },
                "time_dependent_live_births": True,
                "time_dependent_population_fraction": False,
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    def get_configuration(self, builder: Builder) -> ConfigTree:
        """Retrieve the fertility configuration for this component.

        Override the default to use the ``fertility`` configuration key
        rather than the component name.
        """
        return builder.configuration.fertility

    def setup(self, builder: Builder) -> None:
        """Set up the component by loading birth rate data and obtaining the simulant creator.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Raises
        ------
        ValueError
            If ``population.initialization_age_min`` is not zero.
        """
        config = builder.configuration.population
        if config.initialization_age_min != 0:
            raise ValueError(
                "Configuration key 'initialization_age_min' must be 0 if using FertilityCrudeBirthRate. "
                f"Provided value: {config.initialization_age_min}"
            )
        self.birth_rate = self._get_live_births_per_year(builder)

        self.clock = builder.time.clock()
        self.randomness = builder.randomness
        self.simulant_creator = builder.population.get_simulant_creator()

    ##################
    # Helper methods #
    ##################

    def _get_live_births_per_year(self, builder: Builder) -> pd.Series:
        """Compute the simulated number of live births per year.

        Load population structure and live birth data from the configured data
        sources and delegate to :meth:`_compute_live_births_per_year`.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A :class:`pandas.Series` indexed by year containing the expected
            number of new simulant births per year.
        """
        fertility_config = self.get_configuration(builder)

        population_data = self.get_data(
            builder, fertility_config.data_sources.population_structure
        )
        birth_data = self.get_data(builder, fertility_config.data_sources.live_births_by_sex)

        return self._compute_live_births_per_year(builder, population_data, birth_data)

    @staticmethod
    def _compute_live_births_per_year(
        builder: Builder,
        population_data: pd.DataFrame,
        birth_data: pd.DataFrame,
    ) -> pd.Series:
        """Compute the simulated number of live births per year from pre-loaded data.

        Transform population structure and live birth data into a per-year birth
        rate scaled to the simulation's initial population size. Handle
        time-dependent vs. fixed birth rates and population fractions according
        to the fertility configuration, and extend the series to cover the
        simulation end year if needed.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        population_data
            Population structure data with columns including ``age_start``,
            ``age_end``, ``year_start``, ``year_end``, and ``value``.
        birth_data
            Live births by sex data with columns including ``parameter``,
            ``year_start``, and ``value``.

        Returns
        -------
            A :class:`pandas.Series` indexed by year containing the expected
            number of new simulant births per year.
        """
        FertilityCrudeBirthRate._validate_crude_birth_rate_data(
            builder, population_data.year_end.max()
        )
        population_data = FertilityCrudeBirthRate._rescale_final_age_bin(
            builder, population_data
        )

        initial_population_size = builder.configuration.population.population_size
        population_data = population_data.groupby(["year_start"])["value"].sum()
        birth_data = (
            birth_data[birth_data.parameter == "mean_value"]
            .drop(columns=["parameter"])
            .groupby(["year_start"])["value"]
            .sum()
        )

        start_year = builder.configuration.time.start.year
        if (
            builder.configuration.interpolation.extrapolate
            and start_year > birth_data.index.max()
        ):
            start_year = birth_data.index.max()

        if not builder.configuration.fertility.time_dependent_live_births:
            birth_data = birth_data.at[start_year]

        if not builder.configuration.fertility.time_dependent_population_fraction:
            population_data = population_data.at[start_year]

        live_birth_rate = initial_population_size / population_data * birth_data

        if isinstance(live_birth_rate, (int, float)):
            live_birth_rate = pd.Series(
                live_birth_rate,
                index=pd.RangeIndex(
                    builder.configuration.time.start.year,
                    builder.configuration.time.end.year + 1,
                    name="year",
                ),
            )
        else:
            live_birth_rate = (
                live_birth_rate.reset_index()
                .rename(columns={"year_start": "year"})
                .set_index("year")
                .value
            )
            exceeds_data = builder.configuration.time.end.year > live_birth_rate.index.max()
            if exceeds_data:
                new_index = pd.RangeIndex(
                    live_birth_rate.index.min(), builder.configuration.time.end.year + 1
                )
                live_birth_rate = live_birth_rate.reindex(
                    new_index, fill_value=live_birth_rate.at[live_birth_rate.index.max()]
                )
        return live_birth_rate

    @staticmethod
    def _validate_crude_birth_rate_data(builder: Builder, data_year_max: int) -> None:
        """Validate that birth rate data covers the simulation time period."""
        exceeds_data = builder.configuration.time.end.year > data_year_max
        if exceeds_data and not builder.configuration.interpolation.extrapolate:
            raise ValueError("Trying to extrapolate beyond the end of available birth data.")

    @staticmethod
    def _rescale_final_age_bin(
        builder: Builder, population_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Clip and rescale the final age bin to match ``initialization_age_max``."""
        initialization_age_max = builder.configuration.population.to_dict().get(
            "initialization_age_max", None
        )

        if initialization_age_max:
            population_data = population_data.loc[
                population_data["age_start"] < initialization_age_max
            ].copy()
            cut_bin_idx = initialization_age_max <= population_data["age_end"]
            cut_age_start = population_data.loc[cut_bin_idx, "age_start"]
            cut_age_end = population_data.loc[cut_bin_idx, "age_end"]
            population_data.loc[cut_bin_idx, "value"] *= (
                initialization_age_max - cut_age_start
            ) / (cut_age_end - cut_age_start)
            population_data.loc[cut_bin_idx, "age_end"] = initialization_age_max
        return population_data

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """Add new simulants based on :term:`crude birth rate <Crude Birth Rate>` and a Poisson distribution.

        Parameters
        ----------
        event
            The event that triggered this method call.
        """
        birth_rate = self.birth_rate.at[self.clock().year]
        step_size = utilities.to_years(event.step_size)

        mean_births = birth_rate * step_size
        # Assume births occur as a Poisson process
        r = np.random.RandomState(seed=self.randomness.get_seed("crude_birth_rate"))
        simulants_to_add = r.poisson(mean_births)

        if simulants_to_add > 0:
            self.simulant_creator(
                simulants_to_add,
                {
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                },
            )


class FertilityAgeSpecificRates(Component):
    """Simulant-specific model of fertility based on :term:`age-specific fertility rates <Age-Specific Fertility Rate>`.

    At each time step, this component determines which living female simulants
    give birth. Eligibility requires at least nine months to have elapsed since
    the simulant's last birth. Newborns are added to the state table with a
    reference to their parent simulant.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, dict]:
        """The default configuration values for this component."""
        return {
            self.name: {
                "data_sources": {
                    "age_specific_fertility_rate": self.load_age_specific_fertility_rate_data
                }
            },
        }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        """Set up the common randomness stream and age-specific fertility lookup tables.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        fertility_rate = self.build_lookup_table(builder, "age_specific_fertility_rate")
        builder.value.register_rate_producer("fertility_rate", source=fertility_rate)

        self.randomness = builder.randomness.get_stream("fertility")
        self.simulant_creator = builder.population.get_simulant_creator()

        builder.population.register_initializer(
            initializer=self.initialize_birth_time_and_parent_id,
            columns=["last_birth_time", "parent_id"],
            required_resources=["sex"],
        )

    #################
    # Setup methods #
    #################

    def load_age_specific_fertility_rate_data(self, builder: Builder) -> pd.DataFrame:
        """Load and filter age-specific fertility rate data from the artifact.

        Reads the ``covariate.age_specific_fertility_rate.estimate`` dataset, retains
        only female mean-value estimates, and returns the relevant columns.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            A :class:`pandas.DataFrame` with columns ``year_start``, ``year_end``,
            ``age_start``, ``age_end``, and ``value``.
        """
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate")
        columns = ["year_start", "year_end", "age_start", "age_end", "value"]
        asfr_data = asfr_data.loc[
            (asfr_data.sex == "Female") & (asfr_data.parameter == "mean_value")
        ].reset_index()[columns]
        return asfr_data

    ########################
    # Event-driven methods #
    ########################

    def initialize_birth_time_and_parent_id(self, pop_data: SimulantData) -> None:
        """Add 'last_birth_time' and 'parent' columns to the state table.

        Parameters
        ----------
        pop_data
            Metadata about the simulants being initialized.
        """
        females = self.population_view.get_filtered_index(pop_data.index, "sex == 'Female'")

        if pop_data.user_data["sim_state"] == "setup":
            parent_id = -1
        else:  # 'sim_state' == 'time_step'
            parent_id = pop_data.user_data["parent_ids"]
        pop_update = pd.DataFrame(
            {"last_birth_time": pd.NaT, "parent_id": parent_id}, index=pop_data.index
        )
        # FIXME: This is a misuse of the column and makes it invalid for
        #    tracking metrics.
        # Do the naive thing, set so all females can have children
        # and none of them have had a child in the last year.
        pop_update.loc[females, "last_birth_time"] = pop_data.creation_time - pd.Timedelta(
            days=utilities.DAYS_PER_YEAR
        )

        self.population_view.initialize(pop_update)

    def on_time_step(self, event: Event) -> None:
        """Produce new children and update parent status on time steps.

        Parameters
        ----------
        event
            The event that triggered this method call.
        """
        # Get a view on all living females who haven't had a child in at least nine months.
        nine_months_ago = pd.Timestamp(event.time - PREGNANCY_DURATION)
        last_birth_time = self.population_view.get(
            event.index,
            "last_birth_time",
            query="is_alive == True and sex == 'Female'",
        )
        eligible_females_idx = last_birth_time[last_birth_time < nine_months_ago].index
        fertility_rate = self.population_view.get(eligible_females_idx, "fertility_rate")
        had_children_idx = self.randomness.filter_for_rate(
            fertility_rate.index, fertility_rate
        )
        self.population_view.update(
            "last_birth_time",
            lambda _: pd.Series(event.time, index=had_children_idx, name="last_birth_time"),
        )

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children_idx)
        if num_babies:
            self.simulant_creator(
                num_babies,
                {
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                    "parent_ids": had_children_idx,
                },
            )
