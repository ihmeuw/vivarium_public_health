"""
========================
"Special" Disease Models
========================

This module contains frequently used, but non-standard disease models.

"""
from __future__ import annotations

import re
from collections import namedtuple
from collections.abc import Callable
from operator import gt, lt
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_public_health.causal_factor.calibration_constant import (
    register_risk_affected_attribute_producer,
)
from vivarium_public_health.disease.state import ExcessMortalityState
from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.risks.calibration_constant import (
    register_risk_affected_attribute_producer,
)
from vivarium_public_health.utilities import EntityString, is_non_zero


class RiskAttributableDisease(ExcessMortalityState):
    """Component to model a disease fully attributed by a risk.

    For some (risk, cause) pairs with population attributable fraction
    equal to 1, the clinical definition of the with condition state
    corresponds to a particular exposure of a risk.

    For example, a diagnosis of ``diabetes_mellitus`` occurs after
    repeated measurements of fasting plasma glucose above 7 mmol/L.
    Similarly, ``protein_energy_malnutrition`` corresponds to a weight
    for height ratio that is more than two standard deviations below
    the WHO guideline median weight for height.  In the Global Burden
    of Disease, this corresponds to a categorical exposure to
    ``child_wasting`` in either ``cat1`` or ``cat2``.

    The definition of the disease in terms of exposure should be provided
    in the ``threshold`` configuration flag.  For risks with continuous
    exposure models, the threshold should be provided as a single
    ``float`` or ``int`` with a proper sign between ">" and "<", implying
    that disease is defined by the exposure level ">" than threshold level
    or, "<" than threshold level, respectively.

    For categorical risks, the threshold should be provided as a
    list of categories. This list contains the categories that indicate
    the simulant is experiencing the condition. For a dichotomous risk
    there will be 2 categories. By convention ``cat1`` is used to
    indicate the with condition state and would be the single item in
    the ``threshold`` setting list.

    In addition to the threshold level, you may configure whether
    there is any mortality associated with this disease with the
    ``mortality`` configuration flag.

    Finally, you may specify whether an individual should "recover"
    from the disease if their exposure level falls outside the
    provided threshold.

    In our provided examples, a person would no longer be experiencing
    ``protein_energy_malnutrition`` if their exposure drift out (or
    changes via an intervention) of the provided exposure categories.
    Having your ``fasting_plasma_glucose`` drop below a provided level
    does not necessarily mean you're no longer diabetic.

    To add this component, you need to initialize it with full cause name
    and full risk name, e.g.,

    RiskAttributableDisease('cause.protein_energy_malnutrition',
                            'risk_factor.child_wasting')

    Configuration defaults should be given as, for the continuous risk factor,

    diabetes_mellitus:
        threshold : ">7"
        mortality : True
        recoverable : False

    For the categorical risk factor,

    protein_energy_malnutrition:
        threshold : ['cat1', 'cat2'] # provide the categories to get PEM.
        mortality : True
        recoverable : True
    """

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return f"risk_attributable_disease.{self.cause.name}"

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Provides default configuration values for this component.

        Configuration structure::

            {component_name}:
                data_sources:
                    raw_disability_weight:
                        Source for disability weight data. Default is the
                        artifact key ``{cause}.disability_weight``.
                    cause_specific_mortality_rate:
                        Source for cause-specific mortality rate data. Default
                        uses ``load_cause_specific_mortality_rate_data`` method
                        which loads from artifact if ``mortality`` is True.
                    excess_mortality_rate:
                        Source for excess mortality rate data. Default uses
                        ``load_excess_mortality_rate_data`` method which loads
                        from artifact if ``mortality`` is True.
                    population_attributable_fraction:
                        Source for PAF data. Default is 0, indicating no
                        mediated effects from other risks.
                threshold: str or list
                    Exposure threshold defining disease state. For continuous
                    risks, provide a string like ``">7"`` or ``"<5"``.
                    For categorical risks, provide a list of categories
                    (e.g., ``['cat1', 'cat2']``).
                mortality: bool
                    Whether this disease has associated mortality. Default
                    is True.
                recoverable: bool
                    Whether simulants can recover from this disease when
                    their exposure falls outside the threshold. Default
                    is True.
        """
        return {
            self.name: {
                "data_sources": {
                    "raw_disability_weight": f"{self.cause}.disability_weight",
                    "cause_specific_mortality_rate": self.load_cause_specific_mortality_rate_data,
                    "excess_mortality_rate": self.load_excess_mortality_rate_data,
                    "population_attributable_fraction": 0,
                },
                "threshold": None,
                "mortality": True,
                "recoverable": True,
            }
        }

    @property
    def state_names(self) -> list[str]:
        """List of names of all states in this disease model."""
        return self._state_names

    @property
    def transition_names(self) -> list[TransitionString]:
        """List of names of all transitions in this disease model."""
        return self._transition_names

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, cause: str, risk: str) -> None:
        """
        Parameters
        ----------
        cause
            The full entity string for the cause (e.g.,
            "cause.protein_energy_malnutrition").
        risk
            The full entity string for the risk (e.g.,
            "risk_factor.child_wasting").
        """
        super().__init__()
        self.cause = EntityString(cause)
        self.risk = EntityString(risk)
        self.state_column = self.cause.name
        self.cause_type = "risk_attributable_disease"
        self.model = self.risk.name
        self.state_id = self.cause.name
        self.diseased_event_time_column = f"{self.cause.name}_event_time"
        self.susceptible_event_time_column = f"susceptible_to_{self.cause.name}_event_time"
        self._state_names = [f"{self.cause.name}", f"susceptible_to_{self.cause.name}"]
        self._transition_names = [
            TransitionString(f"susceptible_to_{self.cause.name}_TO_{self.cause.name}")
        ]

        self.disability_weight_name = f"{self.cause.name}.disability_weight"
        self.excess_mortality_rate_name = f"{self.cause.name}.excess_mortality_rate"
        self.exposure_name = f"{self.risk.name}.exposure"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Perform this component's setup.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        self.recoverable = builder.configuration[self.name].recoverable
        self.adjust_state_and_transitions()
        self.clock = builder.time.clock()

        self.raw_disability_weight_table = self.build_lookup_table(
            builder, "raw_disability_weight"
        )
        self.cause_specific_mortality_rate_table = self.build_lookup_table(
            builder, "cause_specific_mortality_rate"
        )
        self.excess_mortality_rate_table = self.build_lookup_table(
            builder, "excess_mortality_rate"
        )
        if self._has_excess_mortality is None:
            self._has_excess_mortality = is_non_zero(self.excess_mortality_rate_table.data)

        self.population_attributable_fraction_table = self.build_lookup_table(
            builder, "population_attributable_fraction"
        )

        builder.value.register_attribute_producer(
            self.disability_weight_name,
            source=self.compute_disability_weight,
            required_resources=[self.raw_disability_weight_table],
        )
        builder.value.register_attribute_modifier(
            "all_causes.disability_weight", modifier=self.disability_weight_name
        )
        builder.value.register_attribute_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            required_resources=[self.cause_specific_mortality_rate_table],
        )
        register_risk_affected_attribute_producer(
            builder=builder,
            name=self.excess_mortality_rate_name,
            source=self.compute_excess_mortality_rate,
            required_resources=[self.excess_mortality_rate_table],
        )
        builder.value.register_attribute_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            required_resources=[self.excess_mortality_rate_name],
        )

        distribution = builder.data.load(f"{self.risk}.distribution")
        threshold = builder.configuration[self.name].threshold

        self.filter_by_exposure = self.get_exposure_filter(distribution, threshold)

        builder.population.register_initializer(
            initializer=self.initialize_disease,
            columns=[
                self.cause.name,
                self.diseased_event_time_column,
                self.susceptible_event_time_column,
            ],
            required_resources=[self.exposure_name],
        )

    #################
    # Setup methods #
    #################

    def adjust_state_and_transitions(self) -> None:
        """Add recovery transition if the disease is recoverable."""
        if self.recoverable:
            self._transition_names.append(
                TransitionString(f"{self.cause.name}_TO_susceptible_to_{self.cause.name}")
            )

    def load_cause_specific_mortality_rate_data(
        self, builder: Builder
    ) -> float | pd.DataFrame:
        """Load cause-specific mortality rate data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The cause-specific mortality rate data, or 0 if mortality
            is disabled.
        """
        if builder.configuration[self.name].mortality:
            csmr_data = builder.data.load(
                f"cause.{self.cause.name}.cause_specific_mortality_rate"
            )
        else:
            csmr_data = 0
        return csmr_data

    def load_excess_mortality_rate_data(self, builder: Builder) -> float | pd.DataFrame:
        """Load excess mortality rate data.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.

        Returns
        -------
            The excess mortality rate data, or 0 if mortality is disabled.
        """
        if builder.configuration[self.name].mortality:
            emr_data = builder.data.load(f"cause.{self.cause.name}.excess_mortality_rate")
        else:
            emr_data = 0
        return emr_data

    def get_exposure_filter(self, distribution: str, threshold: Any) -> Callable:
        """Build a filter function that identifies simulants with the condition.

        Parameters
        ----------
        distribution
            The risk's exposure distribution type.
        threshold
            The exposure threshold defining the disease state. For
            continuous risks, a string like ">7". For categorical risks,
            a list of category names.

        Returns
        -------
            A function that takes a simulant index and returns a boolean
            series indicating which simulants have the condition.
        """
        if distribution in ["dichotomous", "ordered_polytomous", "unordered_polytomous"]:

            def categorical_filter(index):
                exposure = self.population_view.get(index, self.exposure_name)
                return exposure.isin(threshold)

            filter_function = categorical_filter

        else:  # continuous
            Threshold = namedtuple("Threshold", ["operator", "value"])
            threshold_val = re.findall(r"[-+]?\d*\.?\d+", threshold)

            if len(threshold_val) != 1:
                raise ValueError(
                    f"Your {threshold} is an incorrect threshold format. It should include "
                    f'"<" or ">" along with an integer or float number. Your threshold does not '
                    f"include a number or more than one number."
                )

            allowed_operator = {"<", ">"}
            threshold_op = [s for s in threshold.split(threshold_val[0]) if s]
            #  if threshold_op has more than 1 operators or 0 operator
            if len(threshold_op) != 1 or not allowed_operator.intersection(threshold_op):
                raise ValueError(
                    f"Your {threshold} is an incorrect threshold format. It should include "
                    f'"<" or ">" along with an integer or float number.'
                )

            op = gt if threshold_op[0] == ">" else lt
            threshold = Threshold(op, float(threshold_val[0]))

            def continuous_filter(index):
                exposure = self.population_view.get(index, self.exposure_name)
                return threshold.operator(exposure, threshold.value)

            filter_function = continuous_filter

        return filter_function

    ########################
    # Event-driven methods #
    ########################

    def initialize_disease(self, pop_data: SimulantData) -> None:
        """Initialize disease state for new simulants based on exposure.

        Parameters
        ----------
        pop_data
            Metadata about the simulants being initialized.
        """
        new_pop = pd.DataFrame(
            {
                self.cause.name: f"susceptible_to_{self.cause.name}",
                self.diseased_event_time_column: pd.Series(pd.NaT, index=pop_data.index),
                self.susceptible_event_time_column: pd.Series(pd.NaT, index=pop_data.index),
            },
            index=pop_data.index,
        )
        sick = self.filter_by_exposure(pop_data.index)
        new_pop.loc[sick, self.cause.name] = self.cause.name
        new_pop.loc[
            sick, self.diseased_event_time_column
        ] = self.clock()  # match VPH disease, only set w/ condition

        self.population_view.initialize(new_pop)

    def on_time_step(self, event: Event) -> None:
        """Update disease state based on current exposure levels.

        Parameters
        ----------
        event
            The event that triggered this method call.
        """

        def _update_disease_state(pop: pd.DataFrame) -> pd.DataFrame:
            living_idx = self.population_view.get_filtered_index(
                event.index, query="is_alive == True"
            )
            update = pop.loc[living_idx]
            sick = self.filter_by_exposure(living_idx)
            #  if this is recoverable, anyone who gets lower exposure in the event goes back in to susceptible status.
            if self.recoverable:
                change_to_susceptible = (~sick) & (
                    update[self.cause.name] != f"susceptible_to_{self.cause.name}"
                )
                update.loc[
                    change_to_susceptible, self.susceptible_event_time_column
                ] = event.time
                update.loc[
                    change_to_susceptible, self.cause.name
                ] = f"susceptible_to_{self.cause.name}"
            change_to_diseased = sick & (update[self.cause.name] != self.cause.name)
            update.loc[change_to_diseased, self.diseased_event_time_column] = event.time
            update.loc[change_to_diseased, self.cause.name] = self.cause.name
            return update

        self.population_view.update(
            self.population_view.private_columns, _update_disease_state
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_disability_weight(self, index: pd.Index[int]) -> pd.Series[float]:
        """Get the disability weight associated with this disease.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
            An iterable of disability weights indexed by the
            provided ``index``.
        """
        disability_weight = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        disability_weight.loc[with_condition] = self.raw_disability_weight_table(
            with_condition
        )
        return disability_weight

    def compute_excess_mortality_rate(self, index: pd.Index[int]) -> pd.Series[float]:
        """Get the excess mortality rate associated with this disease.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
            An iterable of excess mortality rates indexed by the
            provided ``index``.
        """
        excess_mortality_rate = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        base_excess_mort = self.excess_mortality_rate_table(with_condition)
        excess_mortality_rate.loc[with_condition] = base_excess_mort
        return excess_mortality_rate

    def adjust_cause_specific_mortality_rate(
        self, index: pd.Index[int], rate: pd.Series[float]
    ) -> pd.Series[float]:
        """Modify the cause-specific mortality rate for the given simulants.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        rate
            The base cause-specific mortality rate.

        Returns
        -------
            The adjusted cause-specific mortality rate.
        """
        return rate + self.cause_specific_mortality_rate_table(index)

    def adjust_mortality_rate(
        self, index: pd.Index[int], rates_df: pd.DataFrame
    ) -> pd.DataFrame:
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
        rate = self.population_view.get(
            index, self.excess_mortality_rate_name, skip_post_processor=True
        )
        rates_df[self.cause.name] = rate
        return rates_df

    ##################
    # Helper methods #
    ##################

    def with_condition(self, index: pd.Index[int]) -> pd.Index[int]:
        """Get the subset of simulants who have this condition.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
            The subset of simulants who are alive and have this condition.
        """
        return self.population_view.get_filtered_index(
            index,
            query=f'is_alive == True and {self.cause.name} == "{self.cause.name}"',
        )
