"""
==================
Results Stratifier
==================

This module contains tools for stratifying observed quantities
by specified characteristics.

"""
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView, SimulantData


class SourceType(Enum):
    COLUMN = "column"
    PIPELINE = "pipeline"
    CLOCK = "clock"


@dataclass
class Source:
    """
    A source of information about simulants used to determine which category
    they belong to for a given stratification level. The source name should be
    the name of the column or pipeline being used as the source. If the source
    is of type clock, the name should be something descriptive and unique.

    """

    name: str
    type: SourceType


@dataclass
class StratificationLevel:
    """
    A level of stratification. Each StratificationLevel represents a set of
    mutually exclusive and collectively exhaustive categories into which
    simulants can be assigned.

    The mapper is a function which is applied to the list of sources and
    assigns each simulant to specific category. It will throw an error if its
    output is not one of the stratification's categories. By default, the
    mapper assumes the StratificationLevel has a single source and returns its
    value as the category.

    A current category getter can be defined if it is known that certain
    categories are not possible during specific times. Removing these
    categories from iteration can provide a performance improvement in these
    cases. The primary use case for the current category getter is for
    stratification by time, and in particular stratification by year. By
    default, this will return all categories.

    """

    name: str
    sources: List[Source]
    categories: Set[str]
    mapper: Callable[[pd.Series], str] = None
    current_categories_getter: Callable[[], Set[str]] = None

    def __post_init__(self):
        self._set_mapper()
        self._set_current_categories_getter()

    @property
    def current_categories(self) -> Set[str]:
        return self.current_categories_getter()

    def _set_mapper(self) -> None:
        """
        Sets the mapper to be the identity mapper if no mapper has been
        provided. Additionally, wraps the mapper so that it will throw a
        ValueError if the mapper produces an output that is not in the set of
        this StratificationLevel's categories.
        """
        name = self.name
        categories = self.categories
        mapper = self.mapper if self.mapper else self._default_mapper

        def wrapped_mapper(row: pd.Series) -> pd.Series:
            category = mapper(row)
            if category not in categories:
                raise ValueError(f"Invalid value '{category}' found in {name}.")
            return pd.Series(category)

        self.mapper = wrapped_mapper

    def _set_current_categories_getter(self) -> None:
        self.current_categories_getter = (
            self.current_categories_getter
            if self.current_categories_getter
            else self._default_current_categories_getter
        )

    ###############################
    # Default getters and mappers #
    ###############################

    # noinspection PyMethodMayBeStatic
    def _default_mapper(self, row: pd.Series) -> str:
        return str(row[0])

    def _default_current_categories_getter(self) -> Set[str]:
        return self.categories


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This component manages the assignment of simulants to groups for the
    purpose of stratification. Each observer component will get a reference to
    this component so that it can properly stratify its output.

    """

    name = "results_stratifier"

    configuration_defaults = {
        "observers": {
            "default": [],
        }
    }

    def __init__(self):
        self.metrics_pipeline_name = "metrics"
        self.tmrle_key = "population.theoretical_minimum_risk_life_expectancy"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        self.clock = builder.time.clock()
        self.default_stratification_levels = self._get_default_stratification_levels(builder)
        self.pipelines = {}
        self.columns_required = {"tracked"}
        self.clock_sources = set()
        self.stratification_levels: Dict[str, StratificationLevel] = {}
        self.stratification_groups: Optional[pd.DataFrame] = None

        self.age_bins = self._get_age_bins(builder)

        self.register_stratifications(builder)
        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)
        self._register_timestep_prepare_listener(builder)

    # noinspection PyMethodMayBeStatic
    def _get_default_stratification_levels(self, builder: Builder) -> Set[str]:
        return set(builder.configuration.observers.default)

    def register_stratifications(self, builder: Builder) -> None:
        """Register each desired stratification with calls to _setup_stratification"""

        start_year = builder.configuration.time.start.year
        end_year = builder.configuration.time.end.year

        self.setup_stratification(
            builder,
            name=ResultsStratifier.YEAR,
            sources=[ResultsStratifier.YEAR_SOURCE],
            categories={str(year) for year in range(start_year, end_year + 1)},
            mapper=self.year_stratification_mapper,
            current_category_getter=self.year_current_categories_getter,
        )

        self.setup_stratification(
            builder,
            name=ResultsStratifier.SEX,
            sources=[ResultsStratifier.SEX_SOURCE],
            categories=ResultsStratifier.SEX_CATEGORIES,
        )

        self.setup_stratification(
            builder,
            name=ResultsStratifier.AGE,
            sources=[ResultsStratifier.AGE_SOURCE],
            categories={age_bin for age_bin in self.age_bins["age_group_name"]},
            mapper=self.age_stratification_mapper,
        )

    # noinspection PyMethodMayBeStatic
    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(list(self.columns_required))

    # noinspection PyMethodMayBeStatic
    def _get_age_bins(self, builder: Builder) -> pd.DataFrame:
        raw_age_bins = builder.data.load("population.age_bins")
        age_start = builder.configuration.population.age_start
        exit_age = builder.configuration.population.exit_age

        age_start_mask = age_start < raw_age_bins["age_end"]
        exit_age_mask = raw_age_bins["age_start"] < exit_age if exit_age else True

        age_bins = raw_age_bins.loc[age_start_mask & exit_age_mask, :]
        return age_bins

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=list(self.columns_required),
            requires_values=[pipeline_name for pipeline_name in self.pipelines],
        )

    def _register_timestep_prepare_listener(self, builder: Builder) -> None:
        builder.event.register_listener(
            "time_step__prepare", self.on_time_step_prepare, priority=0
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        if self.stratification_groups is not None:
            # noinspection PyAttributeOutsideInit
            self.stratification_groups = pd.concat(
                [self.stratification_groups, self._set_stratification_groups(pop_data.index)]
            )

    def on_time_step_prepare(self, event: Event) -> None:
        # noinspection PyAttributeOutsideInit
        self.stratification_groups = self._set_stratification_groups(event.index)

    ##################
    # Public methods #
    ##################

    # todo add caching of stratifications
    def group(
        self, index: pd.Index, include: Iterable[str], exclude: Iterable[str]
    ) -> Iterable[Tuple[str, pd.Series]]:
        """Takes a full population index and yields stratified subgroups.

        Parameters
        ----------
        index
            The index of the population to stratify.
        include
            List of stratifications to add to the default stratifications
        exclude
            List of stratifications to remove from the default stratifications

        Yields
        ------
        Tuple[str, pd.Series]
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        stratification_groups = self.stratification_groups.loc[index]

        for stratification in self._get_current_stratifications(include, exclude):
            stratification_key = self._get_stratification_key(stratification)

            group_mask = pd.Series(True, index=index)
            if not index.empty:
                for level, category in stratification:
                    group_mask &= stratification_groups[level.name] == category

            yield stratification_key, group_mask

    ##################
    # Helper methods #
    ##################

    def setup_stratification(
        self,
        builder: Builder,
        name: str,
        sources: List[Source],
        categories: Set[str],
        mapper: Callable[[pd.Series], str] = None,
        current_category_getter: Callable[[], Set[str]] = None,
    ) -> None:
        """
        Defines the characteristics of a stratification level and ensures that
        each of its sources is available to the ResultsStratifier
        """
        stratification_level = StratificationLevel(
            name, sources, categories, mapper, current_category_getter
        )
        self.stratification_levels[name] = stratification_level

        for source in sources:
            if source.type == SourceType.PIPELINE:
                self.pipelines[source.name] = builder.value.get_value(source.name)
            elif source.type == SourceType.COLUMN:
                self.columns_required.add(source.name)
            elif source.type == SourceType.CLOCK:
                self.clock_sources.add(source.name)
            else:
                raise ValueError(f"Invalid stratification source type '{source.type}'.")

    def _get_current_stratifications(
        self, include: Iterable[str], exclude: Iterable[str]
    ) -> List[Tuple[Tuple[StratificationLevel, str], ...]]:
        """
        Gets all stratification combinations. Returns a List of
        Stratifications. Each Stratification is represented as a Tuple of
        Levels. Each Level is represented as a Tuple of a StratificationLevel
        object and string referring to the specific stratification category.

        If no stratification levels are defined, returns a List with a single empty Tuple
        """
        include = set(include)
        exclude = set(exclude)
        level_names = (self.default_stratification_levels | include) - exclude

        groups = [
            [(level, category) for category in level.current_categories]
            for level_name, level in self.stratification_levels.items()
            if level_name in level_names
        ]
        # Get product of all stratification combinations
        return list(itertools.product(*groups))

    @staticmethod
    def _get_stratification_key(
        stratification: Iterable[Tuple[StratificationLevel, str]]
    ) -> str:
        return (
            "_".join([f"{level[0].name}_{level[1]}" for level in stratification])
            .replace(" ", "_")
            .lower()
        )

    def _set_stratification_groups(self, index: pd.Index) -> pd.DataFrame:
        """Determine each simulant's category for each stratification level"""
        pop = self.population_view.get(index, query='tracked == True and alive == "alive"')
        pipeline_values = [
            pd.Series(pipeline(pop.index), name=name)
            for name, pipeline in self.pipelines.items()
        ]
        clock_values = [
            pd.Series(self.clock(), index=pop.index, name=name) for name in self.clock_sources
        ]
        sources = pd.concat([pop] + pipeline_values + clock_values, axis=1)

        stratification_groups = [
            sources[[source.name for source in stratification_level.sources]]
            .apply(stratification_level.mapper, axis=1)
            .squeeze(axis=1)
            .rename(stratification_level.name)
            for stratification_level in self.stratification_levels.values()
        ]
        return pd.concat(stratification_groups, axis=1)

    ##########################
    # Stratification Details #
    ##########################

    AGE = "age"
    AGE_SOURCE = Source("age", SourceType.COLUMN)

    def age_stratification_mapper(self, row: pd.Series) -> str:
        age_group_mask = (
            self.age_bins["age_start"] <= row[ResultsStratifier.AGE_SOURCE.name]
        ) & (row[ResultsStratifier.AGE_SOURCE.name] < self.age_bins["age_end"])
        return str(self.age_bins.loc[age_group_mask, "age_group_name"].squeeze())

    SEX = "sex"
    SEX_SOURCE = Source("sex", SourceType.COLUMN)
    SEX_CATEGORIES = {"Female", "Male"}

    YEAR = "year"
    YEAR_SOURCE = Source("year", SourceType.CLOCK)

    # noinspection PyMethodMayBeStatic
    def year_stratification_mapper(self, row: pd.Series) -> str:
        return str(row[0].year)

    def year_current_categories_getter(self) -> Set[str]:
        return {str(self.clock().year)}
