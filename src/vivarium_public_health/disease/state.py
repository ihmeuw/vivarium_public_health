"""
==============
Disease States
==============

This module contains tools to manage standard disease states.

"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable, LookupTableData
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.state_machine import State, Transient, Transition, Trigger
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor

from vivarium_public_health.disease.transition import (
    ProportionTransition,
    RateTransition,
    TransitionString,
)
from vivarium_public_health.utilities import is_non_zero


class BaseDiseaseState(State):
    ##############
    # Properties #
    ##############

    @property
    def columns_created(self):
        return [self.event_time_column, self.event_count_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [self.model, "alive"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
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
        side_effect_function: Optional[Callable] = None,
        cause_type: str = "cause",
    ):
        super().__init__(state_id, allow_self_transition)  # becomes state_id
        self.cause_type = cause_type

        self.side_effect_function = side_effect_function

        self.event_time_column = self.state_id + "_event_time"
        self.event_count_column = self.state_id + "_event_count"

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

    def get_initialization_parameters(self) -> Dict[str, Any]:
        """Exclude side effect function and cause type from name and __repr__."""
        initialization_parameters = super().get_initialization_parameters()
        for key in ["side_effect_function", "cause_type"]:
            if key in initialization_parameters.keys():
                del initialization_parameters[key]
        return initialization_parameters

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
        event_time : pandas.Timestamp
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

    def get_transition_names(self) -> List[str]:
        transitions = []
        for trans in self.transition_set.transitions:
            init_state = trans.input_state.name.split(".")[1]
            end_state = trans.output_state.name.split(".")[1]
            transitions.append(TransitionString(f"{init_state}_TO_{end_state}"))
        return transitions

    def add_rate_transition(
        self,
        output: "BaseDiseaseState",
        get_data_functions: Dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
    ) -> RateTransition:
        """Builds a RateTransition from this state to the given state.

        Parameters
        ----------
        output
            The end state after the transition.

        get_data_functions
            map from transition type to the function to pull that transition's data
        triggered


        Returns
        -------
        RateTransition
            The created transition object.
        """
        transition = RateTransition(self, output, get_data_functions, triggered)
        self.add_transition(transition)
        return transition

    def add_proportion_transition(
        self,
        output: "BaseDiseaseState",
        get_data_functions: Dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
    ) -> ProportionTransition:
        """Builds a ProportionTransition from this state to the given state.

        Parameters
        ----------
        output
            The end state after the transition.

        get_data_functions
            map from transition type to the function to pull that transition's data

        Returns
        -------
        RateTransition
            The created transition object.
        """
        if "proportion" not in get_data_functions:
            raise ValueError("You must supply a proportion function.")

        transition = ProportionTransition(self, output, get_data_functions, triggered)
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
        side_effect_function: Optional[Callable] = None,
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
        get_data_functions: Dict[str, Callable] = None,
        **kwargs,
    ) -> RateTransition:
        if get_data_functions is None:
            get_data_functions = {
                "incidence_rate": lambda builder, cause: builder.data.load(
                    f"{self.cause_type}.{cause}.incidence_rate"
                )
            }
        elif "incidence_rate" not in get_data_functions:
            raise ValueError("You must supply an incidence rate function.")
        return super().add_rate_transition(output, get_data_functions, **kwargs)


class SusceptibleState(NonDiseasedState):
    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Optional[Callable] = None,
        cause_type: str = "cause",
    ):
        super().__init__(
            state_id,
            allow_self_transition=allow_self_transition,
            side_effect_function=side_effect_function,
            cause_type=cause_type,
            name_prefix="susceptible_to_",
        )


class RecoveredState(NonDiseasedState):
    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Optional[Callable] = None,
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

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        side_effect_function: Optional[Callable] = None,
        cause_type: str = "cause",
        get_data_functions: Dict[str, Callable] = None,
        cleanup_function: Callable = None,
    ):
        """
        Parameters
        ----------
        state_id
            The name of this state.
        allow_self_transition
            Whether this state allows simulants to remain in the state for
            multiple time-steps
        side_effect_function
            A function to be called when this state is entered.
        cause_type
            The type of cause represented by this state. Either "cause" or "sequela".
        get_data_functions
            A dictionary containing a mapping to functions to retrieve data for
            various state attributes
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

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        self.clock = builder.time.clock()

        prevalence_data = self.load_prevalence_data(builder)
        self.prevalence = self.get_prevalence(builder, prevalence_data)

        birth_prevalence_data = self.load_birth_prevalence_data(builder)
        self.birth_prevalence = self.get_birth_prevalence(builder, birth_prevalence_data)

        dwell_time_data = self.load_dwell_time_data(builder)
        self.dwell_time = self.get_dwell_time_pipeline(builder, dwell_time_data)

        disability_weight_data = self.load_disability_weight_data(builder)
        self.has_disability = is_non_zero(disability_weight_data)
        self.base_disability_weight = self.get_base_disability_weight(
            builder, disability_weight_data
        )

        self.disability_weight = self.get_disability_weight_pipeline(builder)

        builder.value.register_value_modifier(
            "disability_weight", modifier=self.disability_weight
        )

        excess_mortality_data = self.load_excess_mortality_rate_data(builder)
        self.has_excess_mortality = is_non_zero(excess_mortality_data)
        self.base_excess_mortality_rate = self.get_base_excess_mortality_rate(
            builder, excess_mortality_data
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

    def load_prevalence_data(self, builder: Builder) -> LookupTableData:
        if "prevalence" in self._get_data_functions:
            return self._get_data_functions["prevalence"](builder, self.state_id)
        else:
            return builder.data.load(f"{self.cause_type}.{self.state_id}.prevalence")

    def get_prevalence(
        self, builder: Builder, prevalence_data: LookupTableData
    ) -> LookupTable:
        """Builds a LookupTable for the prevalence of this state.

        Parameters
        ----------
        builder
            Interface to access simulation managers.
        prevalence_data
            The data to use to build the LookupTable.

        Returns
        -------
        LookupTable
            The LookupTable for the prevalence of this state.
        """
        return builder.lookup.build_table(
            prevalence_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def load_birth_prevalence_data(self, builder: Builder) -> LookupTableData:
        if "birth_prevalence" in self._get_data_functions:
            return self._get_data_functions["birth_prevalence"](builder, self.state_id)
        else:
            return 0

    def get_birth_prevalence(
        self, builder: Builder, birth_prevalence_data: LookupTableData
    ) -> LookupTable:
        """
        Builds a LookupTable for the birth prevalence of this state.

        Parameters
        ----------
        builder
            Interface to access simulation managers.
        birth_prevalence_data
            The data to use to build the LookupTable.

        Returns
        -------
        LookupTable
            The LookupTable for the birth prevalence of this state.
        """
        return builder.lookup.build_table(
            birth_prevalence_data, key_columns=["sex"], parameter_columns=["year"]
        )

    def load_dwell_time_data(self, builder: Builder) -> LookupTableData:
        if "dwell_time" in self._get_data_functions:
            dwell_time = self._get_data_functions["dwell_time"](builder, self.state_id)
        else:
            dwell_time = 0

        if isinstance(dwell_time, pd.Timedelta):
            dwell_time = dwell_time.total_seconds() / (60 * 60 * 24)
        if (
            isinstance(dwell_time, pd.DataFrame) and np.any(dwell_time.value != 0)
        ) or dwell_time > 0:
            self.transition_set.allow_null_transition = True

        return dwell_time

    def get_dwell_time_pipeline(
        self, builder: Builder, dwell_time_data: LookupTableData
    ) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=builder.lookup.build_table(
                dwell_time_data, key_columns=["sex"], parameter_columns=["age", "year"]
            ),
            requires_columns=["age", "sex"],
        )

    def load_disability_weight_data(self, builder: Builder) -> LookupTableData:
        if "disability_weight" in self._get_data_functions:
            disability_weight = self._get_data_functions["disability_weight"](
                builder, self.state_id
            )
        else:
            disability_weight = builder.data.load(
                f"{self.cause_type}.{self.state_id}.disability_weight"
            )

        if isinstance(disability_weight, pd.DataFrame) and len(disability_weight) == 1:
            disability_weight = disability_weight.value[0]  # sequela only have single value

        return disability_weight

    def get_base_disability_weight(
        self, builder: Builder, disability_weight_data: LookupTableData
    ) -> LookupTable:
        """
        Builds a LookupTable for the base disability weight of this state.

        Parameters
        ----------
        builder
            Interface to access simulation managers.
        disability_weight_data
            The data to use to build the LookupTable.

        Returns
        -------
        LookupTable
            The LookupTable for the disability weight of this state.
        """
        return builder.lookup.build_table(
            disability_weight_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", self.model],
        )

    def load_excess_mortality_rate_data(self, builder: Builder) -> LookupTableData:
        if "excess_mortality_rate" in self._get_data_functions:
            return self._get_data_functions["excess_mortality_rate"](builder, self.state_id)
        elif builder.data.load(f"cause.{self.model}.restrictions")["yld_only"]:
            return 0
        else:
            return builder.data.load(
                f"{self.cause_type}.{self.state_id}.excess_mortality_rate"
            )

    def get_base_excess_mortality_rate(
        self, builder: Builder, excess_mortality_data: LookupTableData
    ) -> LookupTable:
        """
        Builds a LookupTable for the base excess mortality rate of this state.

        Parameters
        ----------
        builder
            Interface to access simulation managers.
        excess_mortality_data
            The data to use to build the LookupTable.

        Returns
        -------
        LookupTable
            The LookupTable for the base excess mortality rate of this state.
        """
        return builder.lookup.build_table(
            excess_mortality_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def get_excess_mortality_rate_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_rate_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=["age", "sex", "alive", self.model],
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

    def add_rate_transition(
        self,
        output: BaseDiseaseState,
        get_data_functions: Dict[str, Callable] = None,
        **kwargs,
    ) -> RateTransition:
        if get_data_functions is None:
            get_data_functions = {
                "remission_rate": lambda builder, cause: builder.data.load(
                    f"{self.cause_type}.{cause}.remission_rate"
                )
            }
        elif (
            "remission_rate" not in get_data_functions
            and "transition_rate" not in get_data_functions
        ):
            raise ValueError("You must supply a transition rate or remission rate function.")
        return super().add_rate_transition(output, get_data_functions, **kwargs)

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
        event_time:
            The time at which this transition occurs.
        population_view:
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
        `pandas.Series`
            An iterable of disability weights indexed by the provided `index`.
        """
        disability_weight = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        disability_weight.loc[with_condition] = self.base_disability_weight(with_condition)
        return disability_weight

    def compute_excess_mortality_rate(self, index: pd.Index) -> pd.Series:
        excess_mortality_rate = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        base_excess_mort = self.base_excess_mortality_rate(with_condition)
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
        rates_df : `pandas.DataFrame`

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

    def _filter_for_transition_eligibility(self, index, event_time):
        """Filter out all simulants who haven't been in the state for the prescribed dwell time.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
        pd.Index
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
