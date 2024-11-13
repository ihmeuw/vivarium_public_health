"""
===================
Mortality Observers
===================

This module contains tools for observing cause-specific and
excess mortality in the simulation, including "other causes".

"""

from typing import Any

import pandas as pd
from layered_config_tree import LayeredConfigTree
from pandas.api.types import CategoricalDtype
from vivarium.framework.engine import Builder

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.results.columns import COLUMNS
from vivarium_public_health.results.observer import PublicHealthObserver
from vivarium_public_health.results.simple_cause import SimpleCause


class MortalityObserver(PublicHealthObserver):
    """An observer for cause-specific deaths and ylls (including "other causes").

    By default, this counts cause-specific deaths and years of life lost over
    the full course of the simulation. It can be configured to add or remove
    stratification groups to the default groups defined by a
    :class:ResultsStratifier. The aggregate configuration key can be set to
    True to aggregate all deaths and ylls into a single observation and remove
    the stratification by cause of death to improve runtime.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            stratification:
                mortality:
                    exclude:
                        - "sex"
                    include:
                        - "sample_stratification"

    This observer needs to access the has_excess_mortality attribute of the causes
    we're observing, but this attribute gets defined in the setup of the cause models.
    As a result, the model specification should list this observer after causes.

    Attributes
    ----------
    required_death_columns
        Columns required by the deaths observation.
    required_yll_columns
        Columns required by the ylls observation.
    clock
        The simulation clock.
    causes_of_death
        Causes of death to be observed.

    """

    def __init__(self) -> None:
        super().__init__()
        self.required_death_columns = ["alive", "exit_time", "cause_of_death"]
        self.required_yll_columns = [
            "alive",
            "cause_of_death",
            "exit_time",
            "years_of_life_lost",
        ]

    ##############
    # Properties #
    ##############

    @property
    def mortality_classes(self) -> list[type]:
        """The classes to be considered as causes of death."""
        return [DiseaseState, RiskAttributableDisease]

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """A dictionary containing the defaults for any configurations managed by
        this component.
        """
        config_defaults = super().configuration_defaults
        config_defaults["stratification"][self.get_configuration_name()]["aggregate"] = False
        return config_defaults

    @property
    def columns_required(self) -> list[str]:
        """Columns required by this observer."""
        return [
            "alive",
            "years_of_life_lost",
            "cause_of_death",
            "exit_time",
        ]

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        """Set up the observer."""
        self.clock = builder.time.clock()
        self.set_causes_of_death(builder)

    def set_causes_of_death(self, builder: Builder) -> None:
        """Set the causes of death to be observed.

        The causes to be observed are any registered components of class types
        found in the ``mortality_classes`` property.

        Notes
        -----
        We do not actually exclude any categories in this method.

        Also note that we add 'not_dead' and 'other_causes' categories here.
        """
        causes_of_death = [
            cause
            for cause in builder.components.get_components_by_type(
                tuple(self.mortality_classes)
            )
            if cause.has_excess_mortality
        ]

        # Convert to SimpleCauses and add on other_causes and not_dead
        self.causes_of_death = [
            SimpleCause.create_from_specific_cause(cause) for cause in causes_of_death
        ] + [
            SimpleCause("not_dead", "not_dead", "cause"),
            SimpleCause("other_causes", "other_causes", "cause"),
        ]

    def get_configuration(self, builder: Builder) -> LayeredConfigTree:
        """Get the stratification configuration for this observer.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The stratification configuration for this observer.
        """
        return builder.configuration.stratification[self.get_configuration_name()]

    def register_observations(self, builder: Builder) -> None:
        """Register stratifications and observations.

        Notes
        -----
        Ideally, each observer registers a single observation. This one, however,
        registeres two.

        While it's typical for all stratification registrations to be encapsulated
        in a single class (i.e. the
        :class:ResultsStratifier <vivarium_public_health.results.stratification.ResultsStratifier),
        this observer potentially registers an additional one. While it could
        be registered in the ``ResultsStratifier`` as well, it is specific to
        this observer and so it is registered here while we have easy access
        to the required categories.
        """
        pop_filter = 'alive == "dead" and tracked == True'
        additional_stratifications = self.configuration.include
        if not self.configuration.aggregate:
            # manually append 'not_dead' as an excluded cause
            excluded_categories = (
                builder.configuration.stratification.excluded_categories.to_dict().get(
                    "cause_of_death", []
                )
            ) + ["not_dead"]
            builder.results.register_stratification(
                "cause_of_death",
                [cause.state_id for cause in self.causes_of_death],
                excluded_categories=excluded_categories,
                requires_columns=["cause_of_death"],
            )
            additional_stratifications += ["cause_of_death"]
        self.register_adding_observation(
            builder=builder,
            name="deaths",
            pop_filter=pop_filter,
            requires_columns=self.required_death_columns,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_deaths,
        )
        self.register_adding_observation(
            builder=builder,
            name="ylls",
            pop_filter=pop_filter,
            requires_columns=self.required_yll_columns,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.calculate_ylls,
        )

    ###############
    # Aggregators #
    ###############

    def count_deaths(self, x: pd.DataFrame) -> float:
        """Count the number of deaths that occurred during this time step."""
        died_of_cause = x["exit_time"] > self.clock()
        return sum(died_of_cause)

    def calculate_ylls(self, x: pd.DataFrame) -> float:
        """Calculate the years of life lost during this time step."""
        died_of_cause = x["exit_time"] > self.clock()
        return x.loc[died_of_cause, "years_of_life_lost"].sum()

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Rename the appropriate column to 'entity'.

        The primary thing this method does is rename the 'cause_of_death' column
        to 'entity' (or, it we are aggregating, and there is no 'cause_of_death'
        column, we simply create a new 'entity' column). We do this here instead
        of the 'get_entity_column' method simply because we do not want the
        'cause_of_death' at all. If we keep it here and then return it as the
        entity column later, the final results would have both.

        Parameters
        ----------
        measure
            The measure.
        results
            The results to format.

        Returns
        -------
            The formatted results.
        """
        results = results.reset_index()
        if self.configuration.aggregate:
            results[COLUMNS.ENTITY] = "all_causes"
        else:
            results.rename(columns={"cause_of_death": COLUMNS.ENTITY}, inplace=True)
        return results

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity_type' column values."""
        entity_type_map = {cause.state_id: cause.cause_type for cause in self.causes_of_death}
        return results[COLUMNS.ENTITY].map(entity_type_map).astype(CategoricalDtype())

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'entity' column values."""
        # The entity col was created in the 'format' method
        return results[COLUMNS.ENTITY]

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'sub_entity' column values."""
        return results[COLUMNS.ENTITY]
