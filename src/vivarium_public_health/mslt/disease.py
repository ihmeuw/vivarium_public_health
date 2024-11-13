"""
==============
Disease Models
==============

This module contains tools for modeling diseases in multi-state lifetable
simulations.

"""

from typing import Any

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class AcuteDisease(Component):
    """This component characterises an acute disease.

    An acute disease has a sufficiently short duration, relative to the
    time-step size, that it is not meaningful to talk about prevalence.
    Instead, it simply contributes an excess mortality rate, and/or a
    disability rate.

    Interventions may affect these rates:

    - `<disease>_intervention.excess_mortality`
    - `<disease>_intervention.yld_rate`

    where `<disease>` is the name as provided to the constructor.

    Attributes
    ----------
    disease
        The disease name (referred to as `<disease>` here).
    excess_mortality
        The excess mortality rate for the disease.
    int_excess_mortality
        The excess mortality rate for the disease in the intervention scenario.
    disability_rate
        The years lost due to disability (YLD) rate for the disease.
    int_disability_rate
        The YLD rate for the disease in the intervention scenario.

    """

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease

    def setup(self, builder: Builder) -> None:
        """Load the morbidity and mortality data."""
        mty_data = builder.data.load(f"acute_disease.{self.disease}.mortality")
        mty_rate = builder.lookup.build_table(
            mty_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        yld_data = builder.data.load(f"acute_disease.{self.disease}.morbidity")
        yld_rate = builder.lookup.build_table(
            yld_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.excess_mortality = builder.value.register_rate_producer(
            f"{self.disease}.excess_mortality", source=mty_rate
        )
        self.int_excess_mortality = builder.value.register_rate_producer(
            f"{self.disease}_intervention.excess_mortality", source=mty_rate
        )
        self.disability_rate = builder.value.register_rate_producer(
            f"{self.disease}.yld_rate", source=yld_rate
        )
        self.int_disability_rate = builder.value.register_rate_producer(
            f"{self.disease}_intervention.yld_rate", source=yld_rate
        )
        builder.value.register_value_modifier("mortality_rate", self.mortality_adjustment)
        builder.value.register_value_modifier("yld_rate", self.disability_adjustment)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def mortality_adjustment(self, index, mortality_rate):
        """Adjust the all-cause mortality rate in the intervention scenario, to
        account for any change in prevalence (relative to the BAU scenario).
        """
        delta = self.int_excess_mortality(index) - self.excess_mortality(index)
        return mortality_rate + delta

    def disability_adjustment(self, index, yld_rate):
        """Adjust the years lost due to disability (YLD) rate in the intervention
        scenario, to account for any change in prevalence (relative to the BAU
        scenario).
        """
        delta = self.int_disability_rate(index) - self.disability_rate(index)
        return yld_rate + delta


class Disease(Component):
    """This component characterises a chronic disease.

    It defines the following rates, which may be affected by interventions:

    - `<disease>_intervention.incidence`
    - `<disease>_intervention.remission`
    - `<disease>_intervention.mortality`
    - `<disease>_intervention.morbidity`

    where `<disease>` is the name as provided to the constructor.

    Attributes
    ----------
    disease
        The disease name (referred to as `<disease>` here).
    clock
        The simulation clock.
    start_year
        The simulation start year.
    simplified_equations
        Whether to use simplified equations for the disease model.
    incidence
        The incidence rate for the disease.
    incidence_intervention
        The incidence rate for the disease in the intervention scenario.
    remission
        The remission rate for the disease.
    excess_mortality
        The excess mortality rate for the disease.
    disability_rate
        The years lost due to disability (YLD) rate for the disease.
    initial_prevalence
        The initial prevalence of the disease.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.disease: {
                "simplified_no_remission_equations": False,
            },
        }

    @property
    def columns_created(self) -> list[str]:
        columns = []
        for scenario in ["", "_intervention"]:
            for rate in ["_S", "_C"]:
                for when in ["", "_previous"]:
                    columns.append(self.disease + rate + scenario + when)
        return columns

    @property
    def columns_required(self) -> list[str] | None:
        return ["age", "sex"]

    @property
    def initialization_requirements(self) -> dict[str, list[str]]:
        return {
            "requires_columns": ["age", "sex"],
            "requires_values": [],
            "requires_streams": [],
        }

    def __init__(self, disease: str):
        super().__init__()
        self.disease = disease

    def setup(self, builder: Builder) -> None:
        """Load the disease prevalence and rates data."""
        data_prefix = "chronic_disease.{}.".format(self.disease)
        bau_prefix = self.disease + "."
        int_prefix = self.disease + "_intervention."

        self.clock = builder.time.clock()
        self.start_year = builder.configuration.time.start.year
        self.simplified_equations = builder.configuration[
            self.disease
        ].simplified_no_remission_equations

        inc_data = builder.data.load(data_prefix + "incidence")
        i = builder.lookup.build_table(
            inc_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.incidence = builder.value.register_rate_producer(
            bau_prefix + "incidence", source=i
        )
        self.incidence_intervention = builder.value.register_rate_producer(
            int_prefix + "incidence", source=i
        )

        rem_data = builder.data.load(data_prefix + "remission")
        r = builder.lookup.build_table(
            rem_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.remission = builder.value.register_rate_producer(
            bau_prefix + "remission", source=r
        )

        mty_data = builder.data.load(data_prefix + "mortality")
        f = builder.lookup.build_table(
            mty_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.excess_mortality = builder.value.register_rate_producer(
            bau_prefix + "excess_mortality", source=f
        )

        yld_data = builder.data.load(data_prefix + "morbidity")
        yld_rate = builder.lookup.build_table(
            yld_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.disability_rate = builder.value.register_rate_producer(
            bau_prefix + "yld_rate", source=yld_rate
        )

        prev_data = builder.data.load(data_prefix + "prevalence")
        self.initial_prevalence = builder.lookup.build_table(
            prev_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        builder.value.register_value_modifier("mortality_rate", self.mortality_adjustment)
        builder.value.register_value_modifier("yld_rate", self.disability_adjustment)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize the test population for which this disease is modeled."""
        C = 1000 * self.initial_prevalence(pop_data.index)
        S = 1000 - C

        pop = pd.DataFrame(
            {
                f"{self.disease}_S": S,
                f"{self.disease}_C": C,
                f"{self.disease}_S_previous": S,
                f"{self.disease}_C_previous": C,
                f"{self.disease}_S_intervention": S,
                f"{self.disease}_C_intervention": C,
                f"{self.disease}_S_intervention_previous": S,
                f"{self.disease}_C_intervention_previous": C,
            },
            index=pop_data.index,
        )

        self.population_view.update(pop)

    def on_time_step_prepare(self, event: Event) -> None:
        """Update the disease status for both the BAU and intervention scenarios."""
        # Do not update the disease status in the first year, the initial data
        # describe the disease state at the end of the year.
        if self.clock().year == self.start_year:
            return
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        idx = pop.index
        S_bau, C_bau = pop[f"{self.disease}_S"], pop[f"{self.disease}_C"]
        S_int = pop[f"{self.disease}_S_intervention"]
        C_int = pop[f"{self.disease}_C_intervention"]

        # Extract all of the required rates *once only*.
        i_bau = self.incidence(idx)
        i_int = self.incidence_intervention(idx)
        r = self.remission(idx)
        f = self.excess_mortality(idx)

        # NOTE: if the remission rate is always zero, which is the case for a
        # number of chronic diseases, we can make some simplifications.
        if np.all(r == 0):
            r = 0
            if self.simplified_equations:
                # NOTE: for the 'mslt_reduce_chd' experiment, this results in a
                # slightly lower HALY gain than that obtained when using the
                # full equations (below).
                new_S_bau = S_bau * np.exp(-i_bau)
                new_S_int = S_int * np.exp(-i_int)
                new_C_bau = C_bau * np.exp(-f) + S_bau - new_S_bau
                new_C_int = C_int * np.exp(-f) + S_int - new_S_int
                pop_update = pd.DataFrame(
                    {
                        f"{self.disease}_S": new_S_bau,
                        f"{self.disease}_C": new_C_bau,
                        f"{self.disease}_S_previous": S_bau,
                        f"{self.disease}_C_previous": C_bau,
                        f"{self.disease}_S_intervention": new_S_int,
                        f"{self.disease}_C_intervention": new_C_int,
                        f"{self.disease}_S_intervention_previous": S_int,
                        f"{self.disease}_C_intervention_previous": C_int,
                    },
                    index=pop.index,
                )
                self.population_view.update(pop_update)
                return

        # Calculate common factors.
        i_bau2 = i_bau**2
        i_int2 = i_int**2
        r2 = r**2
        f2 = f**2
        f_r = f * r
        i_bau_r = i_bau * r
        i_int_r = i_int * r
        i_bau_f = i_bau * f
        i_int_f = i_int * f
        f_plus_r = f + r

        # Calculate convenience terms.
        l_bau = i_bau + f_plus_r
        l_int = i_int + f_plus_r
        q_bau = np.sqrt(i_bau2 + r2 + f2 + 2 * i_bau_r + 2 * f_r - 2 * i_bau_f)
        q_int = np.sqrt(i_int2 + r2 + f2 + 2 * i_int_r + 2 * f_r - 2 * i_int_f)
        w_bau = np.exp(-(l_bau + q_bau) / 2)
        w_int = np.exp(-(l_int + q_int) / 2)
        v_bau = np.exp(-(l_bau - q_bau) / 2)
        v_int = np.exp(-(l_int - q_int) / 2)

        # Identify where the denominators are non-zero.
        nz_bau = q_bau != 0
        nz_int = q_int != 0
        denom_bau = 2 * q_bau
        denom_int = 2 * q_int

        new_S_bau = S_bau.copy()
        new_C_bau = C_bau.copy()
        new_S_int = S_int.copy()
        new_C_int = C_int.copy()

        # Calculate new_S_bau, new_C_bau, new_S_int, new_C_int.
        num_S_bau = 2 * (v_bau - w_bau) * (S_bau * f_plus_r + C_bau * r) + S_bau * (
            v_bau * (q_bau - l_bau) + w_bau * (q_bau + l_bau)
        )
        num_S_int = 2 * (v_int - w_int) * (S_int * f_plus_r + C_int * r) + S_int * (
            v_int * (q_int - l_int) + w_int * (q_int + l_int)
        )
        new_S_bau[nz_bau] = num_S_bau[nz_bau] / denom_bau[nz_bau]
        new_S_int[nz_int] = num_S_int[nz_int] / denom_int[nz_int]

        num_C_bau = -(
            (v_bau - w_bau)
            * (2 * (f_plus_r * (S_bau + C_bau) - l_bau * S_bau) - l_bau * C_bau)
            - (v_bau + w_bau) * q_bau * C_bau
        )
        num_C_int = -(
            (v_int - w_int)
            * (2 * (f_plus_r * (S_int + C_int) - l_int * S_int) - l_int * C_int)
            - (v_int + w_int) * q_int * C_int
        )
        new_C_bau[nz_bau] = num_C_bau[nz_bau] / denom_bau[nz_bau]
        new_C_int[nz_int] = num_C_int[nz_int] / denom_int[nz_int]

        pop_update = pd.DataFrame(
            {
                f"{self.disease}_S": new_S_bau,
                f"{self.disease}_C": new_C_bau,
                f"{self.disease}_S_previous": S_bau,
                f"{self.disease}_C_previous": C_bau,
                f"{self.disease}_S_intervention": new_S_int,
                f"{self.disease}_C_intervention": new_C_int,
                f"{self.disease}_S_intervention_previous": S_int,
                f"{self.disease}_C_intervention_previous": C_int,
            },
            index=pop.index,
        )
        self.population_view.update(pop_update)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def mortality_adjustment(self, index, mortality_rate):
        """Adjust the all-cause mortality rate in the intervention scenario, to
        account for any change in disease prevalence (relative to the BAU
        scenario).
        """
        pop = self.population_view.get(index)

        S, C = pop[f"{self.disease}_S"], pop[f"{self.disease}_C"]
        S_prev, C_prev = pop[f"{self.disease}_S_previous"], pop[f"{self.disease}_C_previous"]
        D, D_prev = 1000 - S - C, 1000 - S_prev - C_prev

        S_int, C_int = (
            pop[f"{self.disease}_S_intervention"],
            pop[f"{self.disease}_C_intervention"],
        )
        S_int_prev, C_int_prev = (
            pop[f"{self.disease}_S_intervention_previous"],
            pop[f"{self.disease}_C_intervention_previous"],
        )
        D_int, D_int_prev = 1000 - S_int - C_int, 1000 - S_int_prev - C_int_prev

        # NOTE: as per the spreadsheet, the denominator is from the same point
        # in time as the term being subtracted in the numerator.
        mortality_risk = (D - D_prev) / (S_prev + C_prev)
        mortality_risk_int = (D_int - D_int_prev) / (S_int_prev + C_int_prev)

        delta = np.log((1 - mortality_risk) / (1 - mortality_risk_int))

        return mortality_rate + delta

    def disability_adjustment(self, index, yld_rate):
        """Adjust the years lost due to disability (YLD) rate in the intervention
        scenario, to account for any change in disease prevalence (relative to
        the BAU scenario).
        """
        pop = self.population_view.get(index)

        S, S_prev = pop[f"{self.disease}_S"], pop[f"{self.disease}_S_previous"]
        C, C_prev = pop[f"{self.disease}_C"], pop[f"{self.disease}_C_previous"]
        S_int, S_int_prev = (
            pop[f"{self.disease}_S_intervention"],
            pop[f"{self.disease}_S_intervention_previous"],
        )
        C_int, C_int_prev = (
            pop[f"{self.disease}_C_intervention"],
            pop[f"{self.disease}_C_intervention_previous"],
        )

        # The prevalence rate is the mean number of diseased people over the
        # year, divided by the mean number of alive people over the year.
        # The 0.5 multipliers in the numerator and denominator therefore cancel
        # each other out, and can be removed.
        prevalence_rate = (C + C_prev) / (S + C + S_prev + C_prev)
        prevalence_rate_int = (C_int + C_int_prev) / (S_int + C_int + S_int_prev + C_int_prev)

        delta = prevalence_rate_int - prevalence_rate
        return yld_rate + self.disability_rate(index) * delta
