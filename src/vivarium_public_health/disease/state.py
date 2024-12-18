"""
==============
Disease States
==============

This module contains tools to manage standard disease states.

"""
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.state_machine import State, Transient, Transition, Trigger
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium.types import DataInput, LookupTableData

from vivarium_public_health.disease.exceptions import DiseaseModelError
from vivarium_public_health.disease.transition import (
    ProportionTransition,
    RateTransition,
    TransitionString,
)
from vivarium_public_health.utilities import get_lookup_columns, is_non_zero


class BaseDiseaseState(State):

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        configuration_defaults = super().configuration_defaults
        additional_defaults = {
            "prevalence": self.prevalence,
            "birth_prevalence": self.birth_prevalence,
        }
        data_sources = {
            **configuration_defaults[self.name]["data_sources"],
            **additional_defaults,
        }
        configuration_defaults[self.name]["data_sources"] = data_sources
        return configuration_defaults

    @property
    def columns_created(self):
        return [self.event_time_column, self.event_count_column]

    @property
    def columns_required(self) -> list[str] | None:
        return [self.model, "alive"]

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": [self.model],
            "requires_values": [],
            "requires_streams": [],
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Callable | None = None,
        cause_type: str = "cause",
    ):
        super().__init__(state_id, allow_self_transition)
        self.cause_type = cause_type

        self.side_effect_function = side_effect_function

        self.event_time_column = self.state_id + "_event_time"
        self.event_count_column = self.state_id + "_event_count"
        self.prevalence = 0.0
        self.birth_prevalence = 0.0

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Adds this state's columns to the simulation state table."""
        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(pop_data.index)

        pop_update = self.get_initial_event_times(pop_data)
        self.population_view.update(pop_update)

    ##################
    # Helper methods #
    ##################

    def get_initialization_parameters(self) -> dict[str, Any]:
        """Exclude side effect function and cause type from name and __repr__."""
        initialization_parameters = super().get_initialization_parameters()
        return {"state_id": initialization_parameters["state_id"]}

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        return pd.DataFrame(
            {self.event_time_column: pd.NaT, self.event_count_column: 0}, index=pop_data.index
        )

    def transition_side_effect(self, index: pd.Index, event_time: pd.Timestamp) -> None:
        """Updates the simulation state and triggers any side effects associated with this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time
            The time at which this transition occurs.
        """
        pop = self.population_view.get(index)
        pop[self.event_time_column] = event_time
        pop[self.event_count_column] += 1
        self.population_view.update(pop)

        if self.side_effect_function is not None:
            self.side_effect_function(index, event_time)

    ##################
    # Public methods #
    ##################

    def get_transition_names(self) -> list[str]:
        transitions = []
        for trans in self.transition_set.transitions:
            init_state = trans.input_state.name.split(".")[1]
            end_state = trans.output_state.name.split(".")[1]
            transitions.append(TransitionString(f"{init_state}_TO_{end_state}"))
        return transitions

    def add_rate_transition(
        self,
        output: "BaseDiseaseState",
        get_data_functions: dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
        transition_rate: DataInput | None = None,
        rate_type: str = "transition_rate",
    ) -> RateTransition:
        """Builds a RateTransition from this state to the given state.

        Parameters
        ----------
        output
            The end state after the transition.
        get_data_functions
            Map from transition type to the function to pull that transition's data.
        triggered
            The trigger for the transition
        transition_rate
            The transition rate source. Can be the data itself, a function to
            retrieve the data, or the artifact key containing the data.
        rate_type
            The type of rate. Can be "incidence_rate", "transition_rate", or
            "remission_rate".

        Returns
        -------
            The created transition object.
        """
        transition = RateTransition(
            input_state=self,
            output_state=output,
            get_data_functions=get_data_functions,
            triggered=triggered,
            transition_rate=transition_rate,
            rate_type=rate_type,
        )
        self.add_transition(transition)
        return transition

    def add_proportion_transition(
        self,
        output: "BaseDiseaseState",
        get_data_functions: dict[str, Callable] | None = None,
        triggered=Trigger.NOT_TRIGGERED,
        proportion: DataInput | None = None,
    ) -> ProportionTransition:
        """Builds a ProportionTransition from this state to the given state.

        Parameters
        ----------
        output
            The end state after the transition.
        get_data_functions
            Map from transition type to the function to pull that transition's data.
        triggered
            The trigger for the transition.
        proportion
            The proportion source.  Can be the data itself, a function to
            retrieve the data, or the artifact key containing the data.

        Returns
        -------
            The created transition object.
        """
        if (
            get_data_functions is None or "proportion" not in get_data_functions
        ) and proportion is None:
            raise ValueError("You must supply a proportion function.")

        transition = ProportionTransition(
            input_state=self,
            output_state=output,
            get_data_functions=get_data_functions,
            triggered=triggered,
            proportion=proportion,
        )
        self.add_transition(transition)
        return transition

    def add_dwell_time_transition(
        self, output: "BaseDiseaseState", triggered=Trigger.NOT_TRIGGERED
    ) -> Transition:
        transition = Transition(self, output, triggered=triggered)
        self.add_transition(transition)
        return transition


class NonDiseasedState(BaseDiseaseState):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Callable | None = None,
        cause_type: str = "cause",
        name_prefix: str = "",
    ):
        if not state_id.startswith(name_prefix):
            state_id = f"{name_prefix}{state_id}"
        super().__init__(
            state_id,
            allow_self_transition=allow_self_transition,
            side_effect_function=side_effect_function,
            cause_type=cause_type,
        )

    ##################
    # Public methods #
    ##################

    def add_rate_transition(
        self,
        output: BaseDiseaseState,
        get_data_functions: dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
        transition_rate: DataInput | None = None,
        **_kwargs,
    ) -> RateTransition:
        if get_data_functions is None and transition_rate is None:
            transition_rate = f"{self.cause_type}.{output.state_id}.incidence_rate"
        return super().add_rate_transition(
            output=output,
            get_data_functions=get_data_functions,
            triggered=triggered,
            transition_rate=transition_rate,
            rate_type="incidence_rate",
        )


class SusceptibleState(NonDiseasedState):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Callable | None = None,
        cause_type: str = "cause",
    ):
        super().__init__(
            state_id,
            allow_self_transition=allow_self_transition,
            side_effect_function=side_effect_function,
            cause_type=cause_type,
            name_prefix="susceptible_to_",
        )

    ##################
    # Public methods #
    ##################

    def has_initialization_weights(self) -> bool:
        return True


class RecoveredState(NonDiseasedState):
    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Callable | None = None,
        cause_type: str = "cause",
    ):
        super().__init__(
            state_id,
            allow_self_transition=allow_self_transition,
            side_effect_function=side_effect_function,
            cause_type=cause_type,
            name_prefix="recovered_from_",
        )


class DiseaseState(BaseDiseaseState):
    """State representing a disease in a state machine model."""

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        configuration_defaults = super().configuration_defaults
        additional_defaults = {
            "prevalence": self._prevalence_source,
            "birth_prevalence": self._birth_prevalence_source,
            "dwell_time": self._dwell_time_source,
            "disability_weight": self._disability_weight_source,
            "excess_mortality_rate": self._excess_modality_rate_source,
        }
        data_sources = {
            **configuration_defaults[self.name]["data_sources"],
            **additional_defaults,
        }
        configuration_defaults[self.name]["data_sources"] = data_sources
        return configuration_defaults

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Callable | None = None,
        cause_type: str = "cause",
        get_data_functions: dict[str, Callable] | None = None,
        cleanup_function: Callable | None = None,
        prevalence: DataInput | None = None,
        birth_prevalence: DataInput | None = None,
        dwell_time: DataInput | None = None,
        disability_weight: DataInput | None = None,
        excess_mortality_rate: DataInput | None = None,
    ):
        """
        Parameters
        ----------
        state_id
            The name of this state.
        allow_self_transition
            Whether this state allows simulants to remain in the state for
            multiple time-steps.
        side_effect_function
            A function to be called when this state is entered.
        cause_type
            The type of cause represented by this state. Either "cause" or "sequela".
        get_data_functions
            A dictionary containing a mapping to functions to retrieve data for
            various state attributes.
        cleanup_function
            The cleanup function.
        prevalence
            The prevalence source. This is used to initialize simulants. Can be
            the data itself, a function to retrieve the data, or the artifact
            key containing the data.
        birth_prevalence
            The birth prevalence source. This is used to initialize newborn
            simulants. Can be the data itself, a function to retrieve the data,
            or the artifact key containing the data.
        dwell_time
            The dwell time source. This is used to determine how long a simulant
            must remain in the state before transitioning. Can be the data
            itself, a function to retrieve the data, or the artifact key
            containing the data.
        disability_weight
            The disability weight source. This is used to calculate the
            disability weight for simulants in this state. Can be the data
            itself, a function to retrieve the data, or the artifact key
            containing the data.
        excess_mortality_rate
            The excess mortality rate source. This is used to calculate the
            excess mortality rate for simulants in this state. Can be the data
            itself, a function to retrieve the data, or the artifact key
            containing the data.
        """

        super().__init__(
            state_id,
            allow_self_transition=allow_self_transition,
            side_effect_function=side_effect_function,
            cause_type=cause_type,
        )

        self.excess_mortality_rate_pipeline_name = f"{self.state_id}.excess_mortality_rate"
        self.excess_mortality_rate_paf_pipeline_name = (
            f"{self.excess_mortality_rate_pipeline_name}.paf"
        )

        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )
        self._cleanup_function = cleanup_function

        if get_data_functions is not None:
            warnings.warn(
                "The argument 'get_data_functions' has been deprecated. Use"
                " cause_specific_mortality_rate instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            for data_type in self._get_data_functions:
                try:
                    data_source = locals()[data_type]
                except KeyError:
                    data_source = None

                if locals()[data_type] is not None:
                    raise DiseaseModelError(
                        f"It is not allowed to pass '{data_type}' both as a"
                        " stand-alone argument and as part of get_data_functions."
                    )

        self._prevalence_source = self.get_prevalence_source(prevalence)
        self._birth_prevalence_source = self.get_birth_prevalence_source(birth_prevalence)
        self._dwell_time_source = self.get_dwell_time_source(dwell_time)
        self._disability_weight_source = self.get_disability_weight_source(disability_weight)
        self._excess_modality_rate_source = self.get_excess_mortality_rate_source(
            excess_mortality_rate
        )

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder
            Interface to several simulation tools.
        """
        super().setup(builder)
        self.clock = builder.time.clock()

        self.dwell_time = self.get_dwell_time_pipeline(builder)
        self.disability_weight = self.get_disability_weight_pipeline(builder)

        builder.value.register_value_modifier(
            "all_causes.disability_weight", modifier=self.disability_weight
        )

        self.has_excess_mortality = is_non_zero(
            self.lookup_tables["excess_mortality_rate"].data
        )
        self.excess_mortality_rate = self.get_excess_mortality_rate_pipeline(builder)
        self.joint_paf = self.get_joint_paf(builder)

        builder.value.register_value_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            requires_values=[self.excess_mortality_rate_pipeline_name],
        )

        self.randomness_prevalence = self.get_randomness_prevalence(builder)

    #################
    # Setup methods #
    #################

    def _get_data_functions_source(self, data_type: str) -> DataInput:
        def data_source(builder: Builder) -> LookupTableData:
            return self._get_data_functions[data_type](builder, self.state_id)

        return data_source

    def get_prevalence_source(self, prevalence: DataInput | None) -> DataInput:
        if "prevalence" in self._get_data_functions:
            return self._get_data_functions_source("prevalence")
        elif prevalence is not None:
            return prevalence
        else:
            return f"{self.cause_type}.{self.state_id}.prevalence"

    def get_birth_prevalence_source(self, birth_prevalence: DataInput | None) -> DataInput:
        if "birth_prevalence" in self._get_data_functions:
            return self._get_data_functions_source("birth_prevalence")
        elif birth_prevalence is not None:
            return birth_prevalence
        else:
            return 0.0

    def get_dwell_time_source(self, dwell_time: DataInput | None) -> DataInput:
        if "dwell_time" in self._get_data_functions:
            dwell_time = self._get_data_functions_source("dwell_time")
        elif dwell_time is None:
            dwell_time = 0.0

        def dwell_time_source(builder: Builder) -> LookupTableData:
            dwell_time_ = self.get_data(builder, dwell_time)
            if isinstance(dwell_time_, pd.Timedelta):
                dwell_time_ = dwell_time_.total_seconds() / (60 * 60 * 24)
            if (
                isinstance(dwell_time_, pd.DataFrame) and np.any(dwell_time_.value != 0)
            ) or dwell_time_ > 0:
                self.transition_set.allow_null_transition = True
            return dwell_time_

        return dwell_time_source

    def get_dwell_time_pipeline(self, builder: Builder) -> Pipeline:
        required_columns = get_lookup_columns([self.lookup_tables["dwell_time"]])
        return builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=self.lookup_tables["dwell_time"],
            requires_columns=required_columns,
        )

    def get_disability_weight_source(self, disability_weight: DataInput | None) -> DataInput:
        if "disability_weight" in self._get_data_functions:
            disability_weight = self._get_data_functions_source("disability_weight")
        elif disability_weight is not None:
            disability_weight = disability_weight
        else:
            disability_weight = f"{self.cause_type}.{self.state_id}.disability_weight"

        def disability_weight_source(builder: Builder) -> LookupTableData:
            disability_weight_ = self.get_data(builder, disability_weight)
            if isinstance(disability_weight_, pd.DataFrame) and len(disability_weight_) == 1:
                # sequela only have single value
                disability_weight_ = disability_weight_.value[0]
            return disability_weight_

        return disability_weight_source

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        lookup_columns = get_lookup_columns([self.lookup_tables["disability_weight"]])
        return builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=lookup_columns + ["alive", self.model],
        )

    def get_excess_mortality_rate_source(
        self, excess_mortality_rate: DataInput | None
    ) -> DataInput:
        if "excess_mortality_rate" in self._get_data_functions:
            excess_mortality_rate = self._get_data_functions_source("excess_mortality_rate")
        elif excess_mortality_rate is None:
            excess_mortality_rate = f"{self.cause_type}.{self.state_id}.excess_mortality_rate"

        def excess_mortality_rate_source(builder: Builder) -> LookupTableData:
            if excess_mortality_rate is not None:
                return self.get_data(builder, excess_mortality_rate)
            elif builder.data.load(f"cause.{self.model}.restrictions")["yld_only"]:
                return 0
            return builder.data.load(
                f"{self.cause_type}.{self.state_id}.excess_mortality_rate"
            )

        return excess_mortality_rate_source

    def get_excess_mortality_rate_pipeline(self, builder: Builder) -> Pipeline:
        lookup_columns = get_lookup_columns([self.lookup_tables["excess_mortality_rate"]])
        return builder.value.register_rate_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=lookup_columns + ["alive", self.model],
            requires_values=[self.excess_mortality_rate_paf_pipeline_name],
        )

    def get_joint_paf(self, builder: Builder) -> Pipeline:
        paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            self.excess_mortality_rate_paf_pipeline_name,
            source=lambda idx: [paf(idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def get_randomness_prevalence(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(f"{self.state_id}_prevalent_cases")

    ##################
    # Public methods #
    ##################

    def has_initialization_weights(self) -> bool:
        return True

    def add_rate_transition(
        self,
        output: BaseDiseaseState,
        get_data_functions: dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
        transition_rate: DataInput | None = None,
        rate_type: str = "transition_rate",
    ) -> RateTransition:
        if get_data_functions is None and transition_rate is None:
            transition_rate = f"{self.cause_type}.{self.state_id}.remission_rate"
            rate_type = "remission_rate"
        return super().add_rate_transition(
            output=output,
            get_data_functions=get_data_functions,
            triggered=triggered,
            transition_rate=transition_rate,
            rate_type=rate_type,
        )

    def add_dwell_time_transition(
        self,
        output: "BaseDiseaseState",
        **kwargs,
    ) -> Transition:
        if "dwell_time" not in self._get_data_functions:
            raise ValueError("You must supply a dwell time function.")

        return super().add_dwell_time_transition(output, **kwargs)

    def next_state(
        self, index: pd.Index, event_time: pd.Timestamp, population_view: PopulationView
    ) -> None:
        """Moves a population among different disease states.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time
            The time at which this transition occurs.
        population_view
            A view of the internal state of the simulation.
        """
        eligible_index = self._filter_for_transition_eligibility(index, event_time)
        return super().next_state(eligible_index, event_time, population_view)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_disability_weight(self, index: pd.Index) -> pd.Series:
        """Gets the disability weight associated with this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
            An iterable of disability weights indexed by the provided `index`.
        """
        disability_weight = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        disability_weight.loc[with_condition] = self.lookup_tables["disability_weight"](
            with_condition
        )
        return disability_weight

    def compute_excess_mortality_rate(self, index: pd.Index) -> pd.Series:
        excess_mortality_rate = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        base_excess_mort = self.lookup_tables["excess_mortality_rate"](with_condition)
        joint_mediated_paf = self.joint_paf(with_condition)
        excess_mortality_rate.loc[with_condition] = base_excess_mort * (
            1 - joint_mediated_paf.values
        )
        return excess_mortality_rate

    def adjust_mortality_rate(self, index: pd.Index, rates_df: pd.DataFrame) -> pd.DataFrame:
        """Modifies the baseline mortality rate for a simulant if they are in this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        rates_df
            A DataFrame of mortality rates.

        Returns
        -------
            The modified DataFrame of mortality rates.
        """
        rate = self.excess_mortality_rate(index, skip_post_processor=True)
        rates_df[self.state_id] = rate
        return rates_df

    ##################
    # Helper methods #
    ##################

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        pop_update = super().get_initial_event_times(pop_data)

        simulants_with_condition = self.population_view.subview([self.model]).get(
            pop_data.index, query=f'{self.model}=="{self.state_id}"'
        )
        if not simulants_with_condition.empty:
            infected_at = self._assign_event_time_for_prevalent_cases(
                simulants_with_condition,
                self.clock(),
                self.randomness_prevalence.get_draw,
                self.dwell_time,
            )
            pop_update.loc[infected_at.index, self.event_time_column] = infected_at

        return pop_update

    def with_condition(self, index: pd.Index) -> pd.Index:
        pop = self.population_view.subview(["alive", self.model]).get(index)
        with_condition = pop.loc[
            (pop[self.model] == self.state_id) & (pop["alive"] == "alive")
        ].index
        return with_condition

    @staticmethod
    def _assign_event_time_for_prevalent_cases(
        infected, current_time, randomness_func, dwell_time_func
    ):
        dwell_time = dwell_time_func(infected.index)
        infected_at = dwell_time * randomness_func(infected.index)
        infected_at = current_time - pd.to_timedelta(infected_at, unit="D")
        return infected_at

    def _filter_for_transition_eligibility(self, index, event_time) -> pd.Index:
        """Filter out all simulants who haven't been in the state for the prescribed dwell time.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time
            The time at which this transition occurs.

        Returns
        -------
            A filtered index of the simulants.
        """
        population = self.population_view.get(index, query='alive == "alive"')
        if np.any(self.dwell_time(index)) > 0:
            state_exit_time = population[self.event_time_column] + pd.to_timedelta(
                self.dwell_time(index), unit="D"
            )
            return population.loc[state_exit_time <= event_time].index
        else:
            return index

    def _cleanup_effect(self, index, event_time):
        if self._cleanup_function is not None:
            self._cleanup_function(index, event_time)


class TransientDiseaseState(BaseDiseaseState, Transient):
    pass
