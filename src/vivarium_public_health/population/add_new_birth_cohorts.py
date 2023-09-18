"""
================
Fertility Models
================

This module contains several different models of fertility.

"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_public_health import utilities
from vivarium_public_health.population.data_transformations import (
    get_live_births_per_year,
)

# TODO: Incorporate better data into gestational model (probably as a separate component)
PREGNANCY_DURATION = pd.Timedelta(days=9 * utilities.DAYS_PER_MONTH)


class FertilityDeterministic(Component):
    """Deterministic model of births."""

    CONFIGURATION_DEFAULTS = {
        "fertility": {
            "number_of_new_simulants_each_year": 1000,
        },
    }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.fractional_new_births = 0
        self.simulants_per_year = (
            builder.configuration.fertility.number_of_new_simulants_each_year
        )

        self.simulant_creator = builder.population.get_simulant_creator()

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """Adds a set number of simulants to the population each time step.

        Parameters
        ----------
        event
            The event that triggered the function call.
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
    """Population-level model of births using crude birth rate.

    The number of births added each time step is calculated as

    new_births = sim_pop_size_t0 * live_births / true_pop_size * step_size

    Where

    sim_pop_size_t0 = the initial simulation population size
    live_births = annual number of live births in the true population
    true_pop_size = the true population size

    This component has configuration flags that determine whether the
    live births and the true population size should vary with time.

    Notes
    -----
    The OECD definition of crude birth rate can be found on their
    `website <https://stats.oecd.org/glossary/detail.asp?ID=490>`_,
    while a more thorough discussion of fertility and
    birth rate models can be found on
    `Wikipedia <https://en.wikipedia.org/wiki/Birth_rate>`_ or in demography
    textbooks.

    """

    CONFIGURATION_DEFAULTS = {
        "fertility": {
            "time_dependent_live_births": True,
            "time_dependent_population_fraction": False,
        }
    }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.birth_rate = get_live_births_per_year(builder)

        self.clock = builder.time.clock()
        self.randomness = builder.randomness
        self.simulant_creator = builder.population.get_simulant_creator()

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """Adds new simulants every time step based on the Crude Birth Rate
        and an assumption that birth is a Poisson process
        Parameters
        ----------
        event
            The event that triggered the function call.
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
    """
    A simulant-specific model for fertility and pregnancies.
    """

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return ["last_birth_time", "parent_id"]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["sex"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": ["sex"],
            "requires_values": [],
            "requires_streams": [],
        }

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        """Setup the common randomness stream and
        age-specific fertility lookup tables.
        Parameters
        ----------
        builder : vivarium.engine.Builder
            Framework coordination object.
        """
        age_specific_fertility_rate = self.load_age_specific_fertility_rate_data(builder)
        fertility_rate = builder.lookup.build_table(
            age_specific_fertility_rate, parameter_columns=["age", "year"]
        )
        self.fertility_rate = builder.value.register_rate_producer(
            "fertility rate", source=fertility_rate, requires_columns=["age"]
        )

        self.randomness = builder.randomness.get_stream("fertility")
        self.simulant_creator = builder.population.get_simulant_creator()

    #################
    # Setup methods #
    #################

    def load_age_specific_fertility_rate_data(self, builder: Builder) -> pd.DataFrame:
        asfr_data = builder.data.load("covariate.age_specific_fertility_rate.estimate")
        columns = ["year_start", "year_end", "age_start", "age_end", "mean_value"]
        asfr_data = asfr_data.loc[asfr_data.sex == "Female"][columns]
        return asfr_data

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Adds 'last_birth_time' and 'parent' columns to the state table."""
        pop = self.population_view.subview(["sex"]).get(pop_data.index)
        women = pop.loc[pop.sex == "Female"].index

        if pop_data.user_data["sim_state"] == "setup":
            parent_id = -1
        else:  # 'sim_state' == 'time_step'
            parent_id = pop_data.user_data["parent_ids"]
        pop_update = pd.DataFrame(
            {"last_birth_time": pd.NaT, "parent_id": parent_id}, index=pop_data.index
        )
        # FIXME: This is a misuse of the column and makes it invalid for
        #    tracking metrics.
        # Do the naive thing, set so all women can have children
        # and none of them have had a child in the last year.
        pop_update.loc[women, "last_birth_time"] = pop_data.creation_time - pd.Timedelta(
            days=utilities.DAYS_PER_YEAR
        )

        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        """Produces new children and updates parent status on time steps.
        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        """
        # Get a view on all living women who haven't had a child in at least nine months.
        nine_months_ago = pd.Timestamp(event.time - PREGNANCY_DURATION)
        population = self.population_view.get(
            event.index, query='alive == "alive" and sex =="Female"'
        )
        can_have_children = population.last_birth_time < nine_months_ago
        eligible_women = population[can_have_children]

        rate_series = self.fertility_rate(eligible_women.index)
        had_children = self.randomness.filter_for_rate(eligible_women, rate_series).copy()

        had_children.loc[:, "last_birth_time"] = event.time
        self.population_view.update(had_children["last_birth_time"])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children)
        if num_babies:
            self.simulant_creator(
                num_babies,
                {
                    "age_start": 0,
                    "age_end": 0,
                    "sim_state": "time_step",
                    "parent_ids": had_children.index,
                },
            )
