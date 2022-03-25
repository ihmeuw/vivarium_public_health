from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.time import Time

from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease


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
    output_name: str = None
    mapper: Callable[[pd.Series], str] = None

    def __post_init__(self):
        def default_mapper(row: pd.Series) -> str:
            category = str(row[0])
            if category not in self.categories:
                raise ValueError(f"Invalid value '{category}' found in {self.name}.")
            return category

        self.output_name = self.output_name if self.output_name is not None else self.name
        self.mapper = self.mapper if self.mapper else default_mapper

    def get_valid_categories(
            self, clock: Callable[[], Time] = None
    ) -> List[Tuple["StratificationLevel", str]]:
        if all([source.type == SourceType.CLOCK for source in self.sources]):
            return [(self, str(clock().year))]
        return [(self, value) for value in self.categories]


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a subcomponent for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    NAME = "results_stratifier"

    AGE = "age"
    SEX = "sex"
    YEAR = "year"
    DEATH_YEAR = "death_year"
    CAUSE_OF_DEATH = "cause_of_death"

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
        # The only thing you should request here are resources necessary for results stratification.
        self.clock = None
        self.default_stratification_levels = self._get_default_stratification_levels(builder)
        self.pipelines = {}
        self.columns_required = []
        self.stratification_levels: Dict[str, StratificationLevel] = {}
        self.stratification_groups: pd.DataFrame = None

        self.causes = self._get_causes(builder)

        self.register_stratifications(builder)
        self.population_view = builder.population.get_view(self.columns_required)

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
        include_configs = [
            config.include for name, config in builder.configuration.observers.items()
            if name != "default"
        ]
        all_stratification_levels = (
            self.default_stratification_levels | set(itertools.chain(*include_configs))
        )

        start_year = builder.configuration.time.start.year
        end_year = builder.configuration.time.end.year
        year_categories = {str(year) for year in range(start_year, end_year + 1)}

        if ResultsStratifier.AGE in all_stratification_levels:
            # TODO use a LookupTable?
            age_source = Source("age", SourceType.COLUMN)

            raw_age_bins = builder.data.load("population.age_bins")
            age_start = builder.configuration.population.age_start
            exit_age = builder.configuration.population.exit_age

            age_start_mask = age_start < raw_age_bins["age_end"]
            exit_age_mask = raw_age_bins["age_start"] < exit_age if exit_age else True

            age_bins = raw_age_bins.loc[age_start_mask & exit_age_mask, :]

            age_categories = {age_bin for age_bin in age_bins["age_group_name"]}

            def mapper(row: pd.Series) -> str:
                age_group_mask = (
                    (age_bins["age_start"] <= row[age_source.name])
                    & (row[age_source.name] < age_bins["age_end"])
                )
                return str(age_bins.loc[age_group_mask, "age_group_name"].squeeze())

            self._setup_stratification(
                builder, ResultsStratifier.AGE, [age_source], age_categories, mapper=mapper
            )

        if ResultsStratifier.SEX in all_stratification_levels:
            self._setup_stratification(
                builder,
                ResultsStratifier.SEX,
                [Source("sex", SourceType.COLUMN)],
                {"Female", "Male"},
            )
        if ResultsStratifier.YEAR in all_stratification_levels:
            # noinspection PyUnusedLocal
            def mapper(row: pd.Series) -> str:
                return str(self.clock().year)

            self._setup_stratification(
                builder, ResultsStratifier.YEAR, [], year_categories, mapper=mapper
            )
        if ResultsStratifier.DEATH_YEAR in all_stratification_levels:
            alive_source = Source("alive", SourceType.COLUMN)
            exit_time_source = Source("exit_time", SourceType.COLUMN)

            def mapper(row: pd.Series) -> str:
                if row[alive_source.name] == "alive":
                    return None
                return str(row[exit_time_source.name].year)

            self._setup_stratification(
                builder,
                ResultsStratifier.DEATH_YEAR,
                [alive_source, exit_time_source],
                year_categories,
                ResultsStratifier.YEAR,
                mapper,
            )
        if ResultsStratifier.CAUSE_OF_DEATH in all_stratification_levels:
            cause_of_death_source = Source("cause_of_death", SourceType.COLUMN)
            cause_categories = self.causes

            def mapper(row: pd.Series) -> str:
                if row[cause_of_death_source.name] == "not_dead":
                    return None
                return str(row[cause_of_death_source.name])

            # todo handle causes of death that include the string "death_due_to" if necessary
            self._setup_stratification(
                builder,
                ResultsStratifier.CAUSE_OF_DEATH,
                [cause_of_death_source],
                cause_categories,
                "due_to",
                mapper,
            )

    # noinspection PyMethodMayBeStatic
    def _get_causes(self, builder: Builder) -> Set[str]:
        # todo can we specify only causes with excess mortality?
        # todo can we specify only causes with disability weight?
        diseases = builder.components.get_components_by_type(
            (DiseaseState, RiskAttributableDisease)
        )
        return {c.state_id for c in diseases} | {"other_causes"}

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
        pop_list = [self.population_view.get(index)] + [
            pd.Series(pipeline(index), name=name) for name, pipeline in self.pipelines.items()
        ]
        pop = pd.concat(pop_list, axis=1)

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

    def group(
            self, pop: pd.DataFrame, include: Set[str], exclude: Set[str]
    ) -> Iterable[Tuple[str, pd.DataFrame]]:
        """Takes the full population and yields stratified subgroups.

        Parameters
        ----------
        pop
            The population to stratify.
        include
            List of stratifications to add to the default stratifications
        exclude
            List of stratifications to remove from the default stratifications

        Yields
        ------
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        index = pop.index.intersection(self.stratification_groups.index)
        pop = pop.loc[index]
        stratification_groups = self.stratification_groups.loc[index]

        for stratification in self._get_all_stratifications(include, exclude):
            stratification_key = self._get_stratification_key(stratification)
            if pop.empty:
                pop_in_group = pop
            else:
                mask = True
                for level, category in stratification:
                    mask &= stratification_groups[level.name] == category
                pop_in_group = pop.loc[mask]
            yield stratification_key, pop_in_group

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
        output_name: str = None,
        mapper: Callable[[Union[pd.Series, ]], str] = None,
    ) -> None:
        stratification_level = StratificationLevel(name, sources, categories, output_name, mapper)
        self.stratification_levels[stratification_level.name] = stratification_level

        for source in stratification_level.sources:
            if source.type == SourceType.PIPELINE:
                self.pipelines[source.name] = builder.value.get_value(source.name)
            elif source.type == SourceType.COLUMN:
                self.columns_required.append(source.name)
            elif source.type == SourceType.CLOCK:
                self.clock = self._get_clock(builder)
            else:
                raise ValueError(f"Invalid stratification source type '{source.type}'.")

    def _get_all_stratifications(
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
            level.get_valid_categories(self.clock)
            for level in [self.stratification_levels[level_name] for level_name in level_names]
        ]
        # Get product of all stratification combinations
        return list(itertools.product(*groups))

    @staticmethod
    def _get_stratification_key(stratification: Iterable[Tuple[StratificationLevel, str]]) -> str:
        # todo manage stratification order
        return (
            "_".join([f'{level[0].output_name}_{level[1]}' for level in stratification])
            .replace(" ", "_")
            .lower()
        )
