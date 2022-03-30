from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Time


class SourceType(Enum):
    COLUMN = "column"
    PIPELINE = "pipeline"
    CLOCK = "clock"


@dataclass
class Source:
    # todo add docstrings
    name: str
    type: SourceType


@dataclass
class StratificationLevel:
    # todo add docstrings
    name: str
    sources: List[Source]
    categories: Set[str]
    mapper: Callable[[pd.Series], str] = None
    current_categories_getter: Callable[[], Set[str]] = None

    def __post_init__(self):
        self.mapper = self.mapper if self.mapper else self._default_mapper
        self.current_categories_getter = (
            self.current_categories_getter if self.current_categories_getter
            else self._default_current_categories_getter
        )

    def get_current_categories(self) -> Set[str]:
        return self.current_categories_getter()

    def _default_mapper(self, row: pd.Series) -> str:
        category = str(row[0])
        if category not in self.categories:
            raise ValueError(f"Invalid value '{category}' found in {self.name}.")
        return category

    def _default_current_categories_getter(self) -> Set[str]:
        return self.categories


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a subcomponent for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    NAME = "results_stratifier"

    configuration_defaults = {
        "default": [],
    }

    def __init__(self):
        self.configuration_defaults = self._get_configuration_defaults()

        self.metrics_pipeline_name = "metrics"
        self.tmrle_key = "population.theoretical_minimum_risk_life_expectancy"

    ##########################
    # Initialization methods #
    ##########################

    # noinspection PyMethodMayBeStatic
    def _get_configuration_defaults(self) -> Dict[str, Dict]:
        return {
            "observers": {
                "default": ResultsStratifier.configuration_defaults["default"]
            }
        }

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return ResultsStratifier.NAME

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Perform this component's setup."""
        self.clock = None
        self.default_stratification_levels = self._get_default_stratification_levels(builder)
        self.pipelines = {}
        self.columns_required = ["tracked"]
        self.clock_sources = set()
        self.stratification_levels: Dict[str, StratificationLevel] = {}
        self.stratification_groups: pd.DataFrame = None

        self.age_bins = self._get_age_bins(builder)

        self.register_stratifications(builder)
        self.population_view = self._get_population_view(builder)

        self._register_timestep_prepare_listener(builder)
        self._register_simulation_end_listener(builder)

    # noinspection PyMethodMayBeStatic
    def _get_clock(self, builder: Builder) -> Callable[[], Time]:
        return builder.time.clock()

    # noinspection PyMethodMayBeStatic
    def _get_default_stratification_levels(self, builder: Builder) -> Set[str]:
        return set(builder.configuration.observers.default)

    def register_stratifications(self, builder: Builder) -> None:
        """Register each desired stratification with calls to _setup_stratification"""

        self._setup_stratification(
            builder,
            name=ResultsStratifier.AGE,
            sources=[ResultsStratifier.AGE_SOURCE],
            categories={age_bin for age_bin in self.age_bins["age_group_name"]},
            mapper=self.age_stratification_mapper,
        )

        self._setup_stratification(
            builder,
            name=ResultsStratifier.SEX,
            sources=[ResultsStratifier.SEX_SOURCE],
            categories=ResultsStratifier.SEX_CATEGORIES,
        )

        start_year = builder.configuration.time.start.year
        end_year = builder.configuration.time.end.year

        self._setup_stratification(
            builder,
            name=ResultsStratifier.YEAR,
            sources=[ResultsStratifier.YEAR_SOURCE],
            categories={str(year) for year in range(start_year, end_year + 1)},
            mapper=self.year_stratification_mapper,
            current_category_getter=self.year_current_categories_getter,
        )

    # noinspection PyMethodMayBeStatic
    def _get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view(self.columns_required)

    # noinspection PyMethodMayBeStatic
    def _get_age_bins(self, builder: Builder) -> pd.DataFrame:
        # TODO use a LookupTable?
        raw_age_bins = builder.data.load("population.age_bins")
        age_start = builder.configuration.population.age_start
        exit_age = builder.configuration.population.exit_age

        age_start_mask = age_start < raw_age_bins["age_end"]
        exit_age_mask = raw_age_bins["age_start"] < exit_age if exit_age else True

        age_bins = raw_age_bins.loc[age_start_mask & exit_age_mask, :]
        return age_bins

    def _register_timestep_prepare_listener(self, builder: Builder) -> None:
        builder.event.register_listener(
            "time_step__prepare", self._set_stratification_groups, priority=0
        )

    def _register_simulation_end_listener(self, builder: Builder) -> None:
        builder.event.register_listener(
            "simulation_end", self._set_stratification_groups, priority=0
        )

    ########################
    # Event-driven methods #
    ########################

    def _set_stratification_groups(self, event: Event) -> None:
        index = event.index
        pipeline_values = [
            pd.Series(pipeline(index), name=name) for name, pipeline in self.pipelines.items()
        ]
        clock_values = [
            pd.Series(self.clock(), index=index, name=name) for name in self.clock_sources
        ]
        pop = pd.concat([self.population_view.get(index)] + pipeline_values + clock_values, axis=1)

        stratification_groups = [
            pop[[source.name for source in stratification_level.sources]]
            .apply(stratification_level.mapper, axis=1)
            .rename(stratification_level.name)
            for stratification_level in self.stratification_levels.values()
        ]
        self.stratification_groups = pd.concat(stratification_groups, axis=1)

    ##################
    # Public methods #
    ##################

    # todo add caching of stratifications
    def group(
            self, index: pd.Index, include: Set[str], exclude: Set[str]
    ) -> Iterable[Tuple[str, pd.Index]]:
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
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        index = index.intersection(self.stratification_groups.index)
        stratification_groups = self.stratification_groups.loc[index]

        for stratification in self._get_current_stratifications(include, exclude):
            stratification_key = self._get_stratification_key(stratification)
            if index.empty:
                group_index = index
            else:
                mask = True
                for level, category in stratification:
                    mask &= stratification_groups[level.name] == category
                group_index = stratification_groups[mask].index
            yield stratification_key, group_index

    # todo should be able to remove this
    @staticmethod
    def update_labels(
        measure_data: Dict[str, float], label: str
    ) -> Dict[str, float]:
        """Updates a dict of measure data with stratification labels.

        Parameters
        ----------
        measure_data
            The measure data with unstratified column names.
        label
            The stratification labels. Yielded along with the population
            subgroup the measure data was produced from by a call to
            :obj:`ResultsStratifier.group`.

        Returns
        -------
            The measure data with column names updated with the stratification
            labels.

        """
        stratification_label = f"_{label}" if label else ""
        measure_data = {f"{k}{stratification_label}": v for k, v in measure_data.items()}
        return measure_data

    ##################
    # Helper methods #
    ##################

    def _setup_stratification(
        self,
        builder: Builder,
        name: str,
        sources: List[Source],
        categories: Set[str],
        mapper: Callable[[Union[pd.Series, ]], str] = None,
        current_category_getter: Callable[[], Set[str]] = None,
    ) -> None:
        stratification_level = StratificationLevel(
            name, sources, categories, mapper, current_category_getter
        )
        self.stratification_levels[stratification_level.name] = stratification_level

        for source in stratification_level.sources:
            if source.type == SourceType.PIPELINE:
                self.pipelines[source.name] = builder.value.get_value(source.name)
            elif source.type == SourceType.COLUMN:
                self.columns_required.append(source.name)
            elif source.type == SourceType.CLOCK:
                self.clock = self._get_clock(builder)
                self.clock_sources.add(source.name)
            else:
                raise ValueError(f"Invalid stratification source type '{source.type}'.")

    def _get_current_stratifications(
            self, include: Set[str], exclude: Set[str]
    ) -> List[Tuple[Tuple[StratificationLevel, str], ...]]:
        """
        Gets all stratification combinations. Returns a List of Stratifications. Each Stratification
        is represented as a Tuple of Levels. Each Level is represented as a Dictionary with keys
        'level' and 'category'. 'level' refers to a StratificationLevel object, and 'category'
        refers to the specific stratification category.

        If no stratification levels are defined, returns a List with a single empty Tuple
        """
        level_names = (self.default_stratification_levels | include) - exclude
        # todo catch KeyError and re-raise more informative error
        groups = [
            [(level, category) for category in level.get_current_categories()]
            for level in [self.stratification_levels[level_name] for level_name in level_names]
        ]
        # Get product of all stratification combinations
        return list(itertools.product(*groups))

    @staticmethod
    def _get_stratification_key(stratification: Iterable[Tuple[StratificationLevel, str]]) -> str:
        # todo manage stratification order
        return (
            "_".join([f'{level[0].name}_{level[1]}' for level in stratification])
            .replace(" ", "_")
            .lower()
        )

    ####################################
    # Standard Stratifications Details #
    ####################################

    AGE = "age"
    AGE_SOURCE = Source("age", SourceType.COLUMN)

    def age_stratification_mapper(self, row: pd.Series) -> str:
        age_group_mask = (
            (self.age_bins["age_start"] <= row[ResultsStratifier.AGE_SOURCE.name])
            & (row[ResultsStratifier.AGE_SOURCE.name] < self.age_bins["age_end"])
        )
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
