"""
========================
"Special" Disease Models
========================

This module contains frequently used, but non-standard disease models.

"""

import re
from collections import namedtuple
from operator import gt, lt
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.values import list_combiner, union_post_processor

from vivarium_public_health.disease.transition import TransitionString
from vivarium_public_health.utilities import EntityString, get_lookup_columns, is_non_zero


class RiskAttributableDisease(Component):
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
    def columns_created(self) -> list[str]:
        return [
            self.cause.name,
            self.diseased_event_time_column,
            self.susceptible_event_time_column,
        ]

    @property
    def columns_required(self) -> list[str] | None:
        return ["alive"]

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": [],
            "requires_values": [f"{self.risk.name}.exposure"],
            "requires_streams": [],
        }

    @property
    def state_names(self):
        return self._state_names

    @property
    def transition_names(self):
        return self._transition_names

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, cause: str, risk: str):
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

        self.excess_mortality_rate_pipeline_name = f"{self.cause.name}.excess_mortality_rate"
        self.excess_mortality_rate_paf_pipeline_name = (
            f"{self.excess_mortality_rate_pipeline_name}.paf"
        )

    # noinspection PyAttributeOutsideInit
    def setup(self, builder):
        self.recoverable = builder.configuration[self.name].recoverable
        self.adjust_state_and_transitions()
        self.clock = builder.time.clock()

        self.disability_weight = builder.value.register_value_producer(
            f"{self.cause.name}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=get_lookup_columns(
                [self.lookup_tables["raw_disability_weight"]]
            ),
        )
        builder.value.register_value_modifier(
            "all_causes.disability_weight", modifier=self.disability_weight
        )
        builder.value.register_value_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            requires_columns=get_lookup_columns(
                [self.lookup_tables["cause_specific_mortality_rate"]]
            ),
        )
        self.has_excess_mortality = is_non_zero(self.lookup_tables["excess_mortality_rate"])

        self.excess_mortality_rate = builder.value.register_value_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=get_lookup_columns(
                [self.lookup_tables["excess_mortality_rate"]]
            ),
            requires_values=[self.excess_mortality_rate_paf_pipeline_name],
        )
        self.joint_paf = builder.value.register_value_producer(
            self.excess_mortality_rate_paf_pipeline_name,
            source=lambda idx: [self.lookup_tables["population_attributable_fraction"](idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        builder.value.register_value_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            requires_values=[self.excess_mortality_rate_pipeline_name],
        )

        distribution = builder.data.load(f"{self.risk}.distribution")
        exposure_pipeline = builder.value.get_value(f"{self.risk.name}.exposure")
        threshold = builder.configuration[self.name].threshold

        self.filter_by_exposure = self.get_exposure_filter(
            distribution, exposure_pipeline, threshold
        )

    #################
    # Setup methods #
    #################

    def adjust_state_and_transitions(self):
        if self.recoverable:
            self._transition_names.append(
                TransitionString(f"{self.cause.name}_TO_susceptible_to_{self.cause.name}")
            )

    def load_cause_specific_mortality_rate_data(self, builder):
        if builder.configuration[self.name].mortality:
            csmr_data = builder.data.load(
                f"cause.{self.cause.name}.cause_specific_mortality_rate"
            )
        else:
            csmr_data = 0
        return csmr_data

    def load_excess_mortality_rate_data(self, builder):
        if builder.configuration[self.name].mortality:
            emr_data = builder.data.load(f"cause.{self.cause.name}.excess_mortality_rate")
        else:
            emr_data = 0
        return emr_data

    def get_exposure_filter(self, distribution, exposure_pipeline, threshold):
        if distribution in ["dichotomous", "ordered_polytomous", "unordered_polytomous"]:

            def categorical_filter(index):
                exposure = exposure_pipeline(index)
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
                exposure = exposure_pipeline(index)
                return threshold.operator(exposure, threshold.value)

            filter_function = continuous_filter

        return filter_function

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
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

        self.population_view.update(new_pop)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index, query='alive == "alive"')
        sick = self.filter_by_exposure(pop.index)
        #  if this is recoverable, anyone who gets lower exposure in the event goes back in to susceptible status.
        if self.recoverable:
            change_to_susceptible = (~sick) & (
                pop[self.cause.name] != f"susceptible_to_{self.cause.name}"
            )
            pop.loc[change_to_susceptible, self.susceptible_event_time_column] = event.time
            pop.loc[
                change_to_susceptible, self.cause.name
            ] = f"susceptible_to_{self.cause.name}"
        change_to_diseased = sick & (pop[self.cause.name] != self.cause.name)
        pop.loc[change_to_diseased, self.diseased_event_time_column] = event.time
        pop.loc[change_to_diseased, self.cause.name] = self.cause.name

        self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def compute_disability_weight(self, index):
        disability_weight = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        disability_weight.loc[with_condition] = self.lookup_tables["raw_disability_weight"](
            with_condition
        )
        return disability_weight

    def compute_excess_mortality_rate(self, index):
        excess_mortality_rate = pd.Series(0.0, index=index)
        with_condition = self.with_condition(index)
        base_excess_mort = self.lookup_tables["excess_mortality_rate"](with_condition)
        joint_mediated_paf = self.joint_paf(with_condition)
        excess_mortality_rate.loc[with_condition] = base_excess_mort * (
            1 - joint_mediated_paf.values
        )
        return excess_mortality_rate

    def adjust_cause_specific_mortality_rate(self, index, rate):
        return rate + self.lookup_tables["cause_specific_mortality_rate"](index)

    def adjust_mortality_rate(self, index, rates_df):
        """Modifies the baseline mortality rate for a simulant if they are in this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        rates_df

        """
        rate = self.excess_mortality_rate(index, skip_post_processor=True)
        rates_df[self.cause.name] = rate
        return rates_df

    ##################
    # Helper methods #
    ##################

    def with_condition(self, index):
        pop = self.population_view.subview(["alive", self.cause.name]).get(index)
        with_condition = pop.loc[
            (pop[self.cause.name] == self.cause.name) & (pop["alive"] == "alive")
        ].index
        return with_condition
