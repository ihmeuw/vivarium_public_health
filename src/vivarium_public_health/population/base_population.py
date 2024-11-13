"""
=========================
The Core Population Model
=========================

This module contains tools for sampling and assigning core demographic
characteristics to simulants.

"""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from layered_config_tree.exceptions import ConfigurationKeyError
from loguru import logger
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream

from vivarium_public_health import utilities
from vivarium_public_health.population.data_transformations import (
    assign_demographic_proportions,
    load_population_structure,
    rescale_binned_proportions,
    smooth_ages,
)


class BasePopulation(Component):
    """Component for producing and aging simulants based on demographic data."""

    CONFIGURATION_DEFAULTS = {
        "population": {
            "initialization_age_min": 0,
            "initialization_age_max": 125,
            "untracking_age": None,
            "include_sex": "Both",  # Either Female, Male, or Both
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> list[str]:
        return ["age", "sex", "alive", "location", "entrance_time", "exit_time"]

    @property
    def time_step_priority(self) -> int:
        return 8

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self._sub_components.append(AgeOutSimulants())

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.population
        self.key_columns = builder.configuration.randomness.key_columns
        if self.config.include_sex not in ["Male", "Female", "Both"]:
            raise ValueError(
                "Configuration key 'population.include_sex' must be one "
                "of ['Male', 'Female', 'Both']. "
                f"Provided value: {self.config.include_sex}."
            )

        # TODO: Remove this when we remove deprecated keys.
        # Validate configuration for deprecated keys
        self._validate_config_for_deprecated_keys()

        source_population_structure = self._load_population_structure(builder)
        self.demographic_proportions = assign_demographic_proportions(
            source_population_structure,
            include_sex=self.config.include_sex,
        )
        self.randomness = self.get_randomness_streams(builder)
        self.register_simulants = builder.randomness.register_simulants

    #################
    # Setup methods #
    #################

    @staticmethod
    def get_randomness_streams(builder: Builder) -> dict[str, RandomnessStream]:
        return {
            "general_purpose": builder.randomness.get_stream("population_generation"),
            "bin_selection": builder.randomness.get_stream(
                "bin_selection", initializes_crn_attributes=True
            ),
            "age_smoothing": builder.randomness.get_stream(
                "age_smoothing", initializes_crn_attributes=True
            ),
            "age_smoothing_age_bounds": builder.randomness.get_stream(
                "age_smoothing_age_bounds", initializes_crn_attributes=True
            ),
        }

    ########################
    # Event-driven methods #
    ########################

    # TODO: Move most of this docstring to an rst file.
    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Creates a population with fundamental demographic and simulation properties.

        When the simulation framework creates new simulants (essentially producing a new
        set of simulant ids) and this component is being used, the newly created simulants
        arrive here first and are assigned the demographic qualities 'age', 'sex',
        and 'location' in a way that is consistent with the demographic distributions
        represented by the population-level data.  Additionally, the simulants are assigned
        the simulation properties 'alive', 'entrance_time', and 'exit_time'.

        The 'alive' parameter is alive or dead.
        In general, most simulation components (except for those computing summary statistics)
        ignore simulants if they are not in the 'alive' category. The 'entrance_time' and
        'exit_time' categories simply mark when the simulant enters or leaves the simulation,
        respectively.  Here we are agnostic to the methods of entrance and exit (e.g., birth,
        migration, death, etc.) as these characteristics can be inferred from this column and
        other information about the simulant and the simulation parameters.
        """

        age_params = {
            "age_start": pop_data.user_data.get(
                "age_start", self.config.initialization_age_min
            ),
            "age_end": pop_data.user_data.get("age_end", self.config.initialization_age_max),
        }

        demographic_proportions = self.get_demographic_proportions_for_creation_time(
            self.demographic_proportions, pop_data.creation_time.year
        )
        self.population_view.update(
            generate_population(
                simulant_ids=pop_data.index,
                creation_time=pop_data.creation_time,
                step_size=pop_data.creation_window,
                age_params=age_params,
                demographic_proportions=demographic_proportions,
                randomness_streams=self.randomness,
                register_simulants=self.register_simulants,
                key_columns=self.key_columns,
            )
        )

    def on_time_step(self, event: Event) -> None:
        """Ages simulants each time step."""
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population["age"] += utilities.to_years(event.step_size)
        self.population_view.update(population)

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def get_demographic_proportions_for_creation_time(
        demographic_proportions, year: int
    ) -> pd.DataFrame:
        reference_years = sorted(set(demographic_proportions.year_start))
        ref_year_index = np.digitize(year, reference_years).item() - 1
        return demographic_proportions[
            demographic_proportions.year_start == reference_years[ref_year_index]
        ]

    # TODO: Remove this method when we remove the deprecated keys
    def _validate_config_for_deprecated_keys(self) -> None:
        mapper = {
            "age_start": "initialization_age_min",
            "age_end": "initialization_age_max",
            "exit_age": "untracking_age",
        }
        deprecated_keys = set(mapper.keys()).intersection(self.config.keys())
        for key in deprecated_keys:
            provided_new_key = False
            for layer in ["override", "model_override"]:
                try:
                    new_key_value = self.config.get_from_layer(mapper[key], layer=layer)
                    provided_new_key = True
                    break
                except ConfigurationKeyError:
                    pass

            if provided_new_key and self.config[key] != new_key_value:
                raise ValueError(
                    f"Configuration contains both '{key}' and '{mapper[key]}' with different values. "
                    f"These keys cannot both be provided. '{key}' will soon be deprecated so please "
                    f"use '{mapper[key]}'. "
                )
            logger.warning(
                "FutureWarning: "
                f"Configuration key '{key}' will be deprecated in future versions of Vivarium "
                f"Public Health. Use the new key '{mapper[key]}' instead."
            )

    def _load_population_structure(self, builder: Builder) -> pd.DataFrame:
        return load_population_structure(builder)


class ScaledPopulation(BasePopulation):
    """This component is to be used in place of BasePopulation when all simulants are
    a subset of the total population and need to be rescaled. The base population
    structure is multiplied by a provided scaling factor. This scaling factor
    can be a dataframe passed in or a string that corresponds to an artifact key.
    If providing an artifact key, users can specify that in the configuration file.
    For example:

    .. code-block:: yaml

    components:
        vivarium_public_health:
            population:
                - ScaledPopulation("some.artifact.key")


    """

    def __init__(self, scaling_factor: str | pd.DataFrame):
        super().__init__()
        self.scaling_factor = scaling_factor
        """Set a multiplicative scaling factor for the population structure."""

    def _load_population_structure(self, builder: Builder) -> pd.DataFrame:
        scaling_factor = self.get_data(builder, self.scaling_factor)
        population_structure = load_population_structure(builder)
        if not isinstance(scaling_factor, pd.DataFrame):
            raise ValueError(
                f"Scaling factor must be a pandas DataFrame. Provided value: {scaling_factor}"
            )
        scaling_factor = scaling_factor.set_index(
            [col for col in scaling_factor.columns if col != "value"]
        )
        population_structure = population_structure.set_index(
            [col for col in population_structure.columns if col != "value"]
        )
        scaled_population_structure = (population_structure * scaling_factor).reset_index()

        return scaled_population_structure


class AgeOutSimulants(Component):
    """Component for handling aged-out simulants"""

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> list[str]:
        """A list of the columns this component requires that it did not create."""
        return self._columns_required

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        super().__init__()
        self._columns_required = None

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.population
        if self.config.untracking_age is not None:
            self._columns_required = ["age", "exit_time", "tracked"]

    def on_time_step_cleanup(self, event: Event) -> None:
        if self.config.untracking_age is None:
            return

        population = self.population_view.get(event.index)
        max_age = float(self.config.untracking_age)
        pop = population[(population["age"] >= max_age) & population["tracked"]].copy()
        if len(pop) > 0:
            pop["tracked"] = pd.Series(False, index=pop.index)
            pop["exit_time"] = event.time
            self.population_view.update(pop)


def generate_population(
    simulant_ids: pd.Index,
    creation_time: pd.Timestamp,
    step_size: pd.Timedelta,
    age_params: dict[str, float],
    demographic_proportions: pd.DataFrame,
    randomness_streams: dict[str, RandomnessStream],
    register_simulants: Callable[[pd.DataFrame], None],
    key_columns: Iterable[str] = ("entrance_time", "age"),
) -> pd.DataFrame:
    """Produces a random set of simulants sampled from the provided `population_data`.

    Parameters
    ----------
    simulant_ids
        Values to serve as the index in the newly generated simulant DataFrame.
    creation_time
        The simulation time when the simulants are created.
    age_params
        Dictionary with keys
            age_start : Start of an age range
            age_end : End of an age range

        The latter two keys can have values specified to generate simulants over an age range.
    demographic_proportions
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)',
        'P(sex, location | age, year)'.
    randomness_streams
        Source of random number generation within the vivarium common random number framework.
    step_size
        The size of the initial time step.
    register_simulants
        A function to register the new simulants with the CRN framework.
    key_columns
        A list of key columns for random number generation.

    Returns
    -------
        Table with columns
            'entrance_time'
                The `pandas.Timestamp` describing when the simulant entered
                the simulation. Set to `creation_time` for all simulants.
            'exit_time'
                The `pandas.Timestamp` describing when the simulant exited
                the simulation. Set initially to `pandas.NaT`.
            'alive'
                One of 'alive' or 'dead' indicating how the simulation
                interacts with the simulant.
            'age'
                The age of the simulant at the current time step.
            'location'
                The location indicating where the simulant resides.
            'sex'
                Either 'Male' or 'Female'.  The sex of the simulant.
    """
    simulants = pd.DataFrame(
        {
            "entrance_time": pd.Series(creation_time, index=simulant_ids),
            "exit_time": pd.Series(pd.NaT, index=simulant_ids),
            "alive": pd.Series("alive", index=simulant_ids),
        },
        index=simulant_ids,
    )
    age_start = float(age_params["age_start"])
    age_end = float(age_params["age_end"])
    if age_start == age_end:
        return _assign_demography_with_initial_age(
            simulants,
            demographic_proportions,
            age_start,
            step_size,
            randomness_streams,
            register_simulants,
        )
    else:  # age_params['age_start'] is not None and age_params['age_end'] is not None
        return _assign_demography_with_age_bounds(
            simulants,
            demographic_proportions,
            age_start,
            age_end,
            randomness_streams,
            register_simulants,
            key_columns,
        )


def _assign_demography_with_initial_age(
    simulants: pd.DataFrame,
    pop_data: pd.DataFrame,
    initial_age: float,
    step_size: pd.Timedelta,
    randomness_streams: dict[str, RandomnessStream],
    register_simulants: Callable[[pd.DataFrame], None],
) -> pd.DataFrame:
    """Assigns age, sex, and location information to the provided simulants given a fixed age.

    Parameters
    ----------
    simulants
        Table that represents the new cohort of agents being added to the simulation.
    pop_data
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)',
        'P(sex, location | age, year)'
    initial_age
        The age to assign the new simulants.
    randomness_streams
        Source of random number generation within the vivarium common random number framework.
    step_size
        The size of the initial time step.
    register_simulants
        A function to register the new simulants with the CRN framework.

    Returns
    -------
        Table with same columns as `simulants` and with the additional
        columns 'age', 'sex',  and 'location'.
    """
    pop_data = pop_data[
        (pop_data.age_start <= initial_age) & (pop_data.age_end >= initial_age)
    ]

    if pop_data.empty:
        raise ValueError(
            "The age {} is not represented by the population data structure".format(
                initial_age
            )
        )

    age_fuzz = randomness_streams["age_smoothing"].get_draw(
        simulants.index
    ) * utilities.to_years(step_size)
    simulants["age"] = initial_age + age_fuzz
    register_simulants(simulants[["entrance_time", "age"]])

    # Assign a demographically accurate location and sex distribution.
    choices = pop_data.set_index(["sex", "location"])[
        "P(sex, location | age, year)"
    ].reset_index()
    decisions = randomness_streams["general_purpose"].choice(
        simulants.index, choices=choices.index, p=choices["P(sex, location | age, year)"]
    )

    simulants["sex"] = choices.loc[decisions, "sex"].values
    simulants["location"] = choices.loc[decisions, "location"].values

    return simulants


def _assign_demography_with_age_bounds(
    simulants: pd.DataFrame,
    pop_data: pd.DataFrame,
    age_start: float,
    age_end: float,
    randomness_streams: dict[str, RandomnessStream],
    register_simulants: Callable[[pd.DataFrame], None],
    key_columns: Iterable[str] = ("entrance_time", "age"),
) -> pd.DataFrame:
    """Assigns an age, sex, and location to the provided simulants given a range of ages.

    Parameters
    ----------
    simulants
        Table that represents the new cohort of agents being added to the simulation.
    pop_data
        Table with columns 'age', 'age_start', 'age_end', 'sex', 'year',
        'location', 'population', 'P(sex, location, age| year)',
        'P(sex, location | age, year)'
    age_start, age_end
        The start and end of the age range of interest, respectively.
    randomness_streams
        Source of random number generation within the vivarium common random number framework.
    register_simulants
        A function to register the new simulants with the CRN framework.
    key_columns
        A list of key columns for random number generation.

    Returns
    -------
        Table with same columns as `simulants` and with the additional columns
        'age', 'sex',  and 'location'.

    """
    pop_data = rescale_binned_proportions(pop_data, age_start, age_end)
    if pop_data.empty:
        raise ValueError(
            f"The age range ({age_start}, {age_end}) is not represented by the "
            f"population data structure."
        )

    # Assign a demographically accurate age, location, and sex distribution.
    sub_pop_data = pop_data[(pop_data.age_start >= age_start) & (pop_data.age_end <= age_end)]
    choices = sub_pop_data.set_index(["age", "sex", "location"])[
        "P(sex, location, age| year)"
    ].reset_index()
    decisions = randomness_streams["bin_selection"].choice(
        simulants.index, choices=choices.index, p=choices["P(sex, location, age| year)"]
    )
    simulants["age"] = choices.loc[decisions, "age"].values
    simulants["sex"] = choices.loc[decisions, "sex"].values
    simulants["location"] = choices.loc[decisions, "location"].values
    simulants = smooth_ages(
        simulants, pop_data, randomness_streams["age_smoothing_age_bounds"]
    )
    register_simulants(simulants[list(key_columns)])
    return simulants
