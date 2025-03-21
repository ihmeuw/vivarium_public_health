"""
===============
Delayed Effects
===============

This module contains tools to represent delayed effects in a multi-state
lifetable simulation.

"""

from typing import Any

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource


class DelayedRisk(Component):
    """A delayed risk represents an exposure whose impact takes time to come into
    effect (e.g., smoking uptake and cessation).

    The data required by this component are:

    Initial prevalence
        The initial level of exposure and post-exposure.
    Incidence rate
        The rate at which people begin to be exposed. This should be specified
        separately for the BAU and the intervention scenario.
    Remission rate
        The rate at which people stop being exposed. This should be specified
        separately for the BAU and the intervention scenario.
    Relative risk of mortality for currently exposed
        (e.g., currently smoking); and
    Relative risk of mortality for post-exposure
        (e.g., stopped smoking :math:`0..N` years ago).
    Disease-specific relative risks for currently exposed
        (e.g., currently smoking)
    Disease-specific relative risks for post-exposure
        (e.g., stopped smoking :math:`0..N` years ago).

    .. note::

       The relative risks are defined in relation to the pre-exposure
       group (whose relative risks are therefore defined to be :math:`1`).


    The configuration options for this component are:

    ``constant_prevalence`` (boolean, default is ``False``)
        If this is set to ``True``, the remission rate in both the BAU and
        intervention will be kept fixed at 0 (i.e., no remission).
    ``tobacco_tax`` (boolean, default is ``False``)
        If this is set to ``True``, additional scaling effects are applied to
        both the incidence and remission rates.
    ``delay`` (integer, default is ``20``)
        The number of years, after remission, during which relative risks
        decrease back to their baseline values.

    Identify the disease(s) for which this delayed risk will have an effect in
    the simulation configuration. For example, to modify the incidence of CHD
    and stroke, this would look like:

    .. code-block:: yaml

       components:
           mslt_port:
               population:
                   - BasePopulation()
                   - Mortality()
                   - Disability()
               disease:
                   - Disease('CHD')
                   - Disease('Stroke')
               delay:
                   - DelayedRisk('tobacco')
               ...
       configuration:
           tobacco:
               constant_prevalence: False
               tobacco_tax: False
               delay: 20
               affects:
                   # This is where the affected diseases should be listed.
                   CHD:
                   Stroke:
    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.risk: {
                "constant_prevalence": False,
                "tobacco_tax": False,
                "delay": 20,
            },
        }

    @property
    def columns_created(self) -> list[str]:
        return self._bin_names

    @property
    def columns_required(self) -> list[str] | None:
        return ["age", "sex", "population"]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return ["age", "sex", "population"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str):
        """
        Parameters
        ----------
        risk
            The name of the exposure (e.g., ``"tobacco"``).
        """
        super().__init__()
        self.risk = risk
        self._bin_names = []

    def setup(self, builder: Builder) -> None:
        """Configure the delayed risk component.

        This involves loading the required data tables, registering event
        handlers and rate modifiers, and setting up the population view.
        """
        self._bin_names = self.get_bin_names()
        self.config = builder.configuration

        self.start_year = builder.configuration.time.start.year
        self.clock = builder.time.clock()

        # Determine whether smoking prevalence should change over time.
        # The alternative scenario is that there is no remission; all people
        # who begin smoking will continue to smoke.
        self.constant_prevalence = self.config[self.risk]["constant_prevalence"]

        self.tobacco_tax = self.config[self.risk]["tobacco_tax"]

        self.bin_years = int(self.config[self.risk]["delay"])

        # Load the initial prevalence.
        prev_data = pivot_load(builder, f"risk_factor.{self.risk}.prevalence")
        self.initial_prevalence = builder.lookup.build_table(
            prev_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        # Load the incidence rates for the BAU and intervention scenarios.
        inc_data = builder.lookup.build_table(
            pivot_load(builder, f"risk_factor.{self.risk}.incidence"),
            key_columns=["sex"],
            parameter_columns=["age", "year"],
        )
        inc_name = "{}.incidence".format(self.risk)
        inc_int_name = "{}_intervention.incidence".format(self.risk)
        self.incidence = builder.value.register_rate_producer(
            inc_name, source=inc_data, component=self
        )
        self.int_incidence = builder.value.register_rate_producer(
            inc_int_name, source=inc_data, component=self
        )

        # Load the remission rates for the BAU and intervention scenarios.
        rem_df = pivot_load(builder, f"risk_factor.{self.risk}.remission")
        # In the constant-prevalence case, assume there is no remission.
        if self.constant_prevalence:
            rem_df["value"] = 0.0
        rem_data = builder.lookup.build_table(
            rem_df, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        rem_name = "{}.remission".format(self.risk)
        rem_int_name = "{}_intervention.remission".format(self.risk)
        self.remission = builder.value.register_rate_producer(
            rem_name, source=rem_data, component=self
        )
        self.int_remission = builder.value.register_rate_producer(
            rem_int_name, source=rem_data, component=self
        )

        # We apply separate mortality rates to the different exposure bins.
        # This requires having access to the life table mortality rate, and
        # also the relative risks associated with each bin.
        self.acm_rate = builder.value.get_value("mortality_rate")
        mort_rr_data = pivot_load(builder, f"risk_factor.{self.risk}.mortality_relative_risk")
        self.mortality_rr = builder.lookup.build_table(
            mort_rr_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        # Register a modifier for each disease affected by this delayed risk.
        diseases = self.config[self.risk].affects.keys()
        for ix, disease in enumerate(diseases):
            self.register_modifier(builder, disease)

        # Load the disease-specific relative risks for each exposure bin.
        dis_rr_data = pivot_load(builder, f"risk_factor.{self.risk}.disease_relative_risk")

        # Check that the relative risk table includes required columns.
        key_columns = ["age_start", "age_end", "sex", "year_start", "year_end"]
        if set(key_columns) & set(dis_rr_data.columns) != set(key_columns):
            # Fallback option, handle tables that do not define bin edges.
            key_columns = ["age", "sex", "year"]
        if set(key_columns) & set(dis_rr_data.columns) != set(key_columns):
            msg = "Missing index columns for disease-specific relative risks"
            raise ValueError(msg)

        self.dis_rr = {}
        for disease in diseases:
            dis_columns = [c for c in dis_rr_data.columns if c.startswith(disease)]
            dis_keys = [c for c in dis_rr_data.columns if c in key_columns]
            if not dis_columns or not dis_keys:
                msg = "No {} relative risks for disease {}"
                raise ValueError(msg.format(self.risk, disease))
            rr_data = dis_rr_data.loc[:, dis_keys + dis_columns]
            dis_prefix = "{}_".format(disease)
            bau_prefix = "{}.".format(self.risk)
            int_prefix = "{}_intervention.".format(self.risk)
            bau_col = {
                c: c.replace(dis_prefix, bau_prefix).replace("post_", "") for c in dis_columns
            }
            int_col = {
                c: c.replace(dis_prefix, int_prefix).replace("post_", "") for c in dis_columns
            }
            for column in dis_columns:
                # NOTE: avoid SettingWithCopyWarning
                rr_data.loc[:, int_col[column]] = rr_data[column]
            rr_data = rr_data.rename(columns=bau_col)
            self.dis_rr[disease] = builder.lookup.build_table(
                rr_data, key_columns=["sex"], parameter_columns=["age", "year"]
            )

        # Load the effects of a tobacco tax.
        tax_inc = pivot_load(builder, f"risk_factor.{self.risk}.tax_effect_incidence")
        tax_rem = pivot_load(builder, f"risk_factor.{self.risk}.tax_effect_remission")
        self.tax_effect_inc = builder.lookup.build_table(
            tax_inc, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        self.tax_effect_rem = builder.lookup.build_table(
            tax_rem, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        mortality_data = pivot_load(builder, "cause.all_causes.mortality")
        self.tobacco_acmr = builder.value.register_rate_producer(
            "tobacco_acmr",
            source=builder.lookup.build_table(
                mortality_data, key_columns=["sex"], parameter_columns=["age", "year"]
            ),
            component=self,
        )

    #################
    # Setup methods #
    #################

    def get_bin_names(self):
        """Return the bin names for both the BAU and the intervention scenario.

        These names take the following forms:

        ``"name.no"``
            The number of people who have never been exposed.
        ``"name.yes"``
            The number of people currently exposed.
        ``"name.N"``
            The number of people N years post-exposure.

        The final bin is the number of people :math:`\ge N` years
        post-exposure.

        The intervention bin names take the form ``"name_intervention.X"``.
        """
        if self.bin_years == 0:
            delay_bins = [str(0)]
        else:
            delay_bins = [str(s) for s in range(self.bin_years + 2)]
        bins = ["no", "yes"] + delay_bins
        bau_bins = ["{}.{}".format(self.risk, bin) for bin in bins]
        int_bins = ["{}_intervention.{}".format(self.risk, bin) for bin in bins]
        all_bins = bau_bins + int_bins
        return all_bins

    def register_modifier(self, builder: Builder, disease: str) -> None:
        """Register that a disease incidence rate will be modified by this
        delayed risk in the intervention scenario.

        Parameters
        ----------
        builder
            The builder object for the simulation, which provides
            access to event handlers and rate modifiers.
        disease
            The name of the disease whose incidence rate will be
            modified.

        """
        # NOTE: we need to modify different rates for chronic and acute
        # diseases. For now, register modifiers for all possible rates.
        rate_templates = [
            "{}_intervention.incidence",
            "{}_intervention.excess_mortality",
            "{}_intervention.yld_rate",
        ]
        for template in rate_templates:
            rate_name = template.format(disease)
            modifier = lambda ix, rate: self.incidence_adjustment(disease, ix, rate)
            builder.value.register_value_modifier(rate_name, modifier, component=self)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Define the initial distribution of the population across the bins, in
        both the BAU and the intervention scenario.
        """
        # Set all bins to zero, in order to create the required columns.
        pop = pd.DataFrame({}, index=pop_data.index)
        for column in self.get_bin_names():
            pop[column] = 0.0

        # Update the life table, so that we can then obtain a view that
        # includes the population counts.
        self.population_view.update(pop)
        pop = self.population_view.get(pop_data.index)

        # Calculate the absolute prevalence by multiplying the fractional
        # prevalence by the population size for each cohort.
        # NOTE: the number of current smokers is defined at the middle of each
        # year; i.e., it corresponds to the person_years.
        bau_acmr = self.tobacco_acmr.source(pop_data.index)
        bau_probability_of_death = 1 - np.exp(-bau_acmr)
        pop.population *= 1 - 0.5 * bau_probability_of_death

        prev = self.initial_prevalence(pop_data.index).mul(pop["population"], axis=0)
        self.population_view.update(prev)

        # Rename the columns and apply the same initial prevalence for the
        # intervention.
        bau_prefix = "{}.".format(self.risk)
        int_prefix = "{}_intervention.".format(self.risk)
        rename_to = {
            c: c.replace(bau_prefix, int_prefix)
            for c in prev.columns
            if c.startswith(bau_prefix)
        }
        int_prev = prev.rename(columns=rename_to)
        self.population_view.update(int_prev)

    def on_time_step_prepare(self, event: Event) -> None:
        """Account for transitions between bins, and for mortality rates.

        These transitions include:
        - New exposures
        - Cessation of exposure
        - Increased duration of time since exposure
        """
        if self.clock().year == self.start_year:
            return

        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        idx = pop.index
        bau_acmr = self.acm_rate.source(idx)
        inc_rate = self.incidence(idx)
        rem_rate = self.remission(idx)
        int_inc_rate = self.int_incidence(idx)
        int_rem_rate = self.int_remission(idx)

        # Identify the relevant columns for the BAU and intervention.
        bin_cols = self.get_bin_names()
        bau_prefix = "{}.".format(self.risk)
        int_prefix = "{}_intervention.".format(self.risk)
        bau_cols = [c for c in bin_cols if c.startswith(bau_prefix)]
        int_cols = [c for c in bin_cols if c.startswith(int_prefix)]

        # Extract the RR of mortality associated with each exposure level.
        mort_rr = self.mortality_rr(idx)

        # Normalise the survival rate; never-smokers should have a mortality
        # rate that is lower than the ACMR, since current-smokers and
        # previous-smokers have higher RRs of mortality.
        weight_by_initial_prevalence = True
        if weight_by_initial_prevalence:
            # Load the initial exposure distribution, because it will be used
            # to adjust the ACMR.
            prev = self.initial_prevalence(pop.index)
            prev = prev.loc[:, bau_cols]
            # Multiply these fractions by the RR of mortality associated with
            # each exposure level.
            bau_wtd_rr = prev.mul(mort_rr.loc[:, bau_cols])
        else:
            # Calculate the fraction of the population in each exposure level.
            bau_popn = pop.loc[:, bau_cols].sum(axis=1)
            bau_prev = pop.loc[:, bau_cols].divide(bau_popn, axis=0)
            # Multiply these fractions by the RR of mortality associated with
            # each exposure level.
            bau_wtd_rr = bau_prev.mul(mort_rr.loc[:, bau_cols])

        # Sum these terms to obtain the net RR of mortality.
        bau_net_rr = bau_wtd_rr.sum(axis=1)
        # The mortality rate for never-smokers is the population ACMR divided
        # by this net RR of mortality.
        bau_acmr_no = bau_acmr.divide(bau_net_rr)

        # NOTE: adjust the RR *after* calculating the ACMR adjustments, but
        # *before* calculating the survival probability for each exposure
        # level.
        penultimate_cols = [s + str(self.bin_years) for s in [bau_prefix, int_prefix]]
        mort_rr.loc[:, penultimate_cols] = 1.0

        # Calculate the mortality risk for non-smokers.
        bau_surv_no = 1 - np.exp(-bau_acmr_no)
        # Calculate the survival probability for each exposure level:
        #     (1 - mort_risk_non_smokers)^RR
        bau_surv_rate = mort_rr.loc[:, bau_cols].rpow(1 - bau_surv_no, axis=0)
        # Calculate the number of survivors for each exposure level (BAU).
        pop.loc[:, bau_cols] = pop.loc[:, bau_cols].mul(bau_surv_rate)

        # Calculate the number of survivors for each exposure level
        # (intervention).
        # NOTE: we apply the same survival rate to each exposure level for
        # the intervention scenario as we used for the BAU scenario.
        rename_to = {c: c.replace(".", "_intervention.") for c in bau_surv_rate.columns}
        int_surv_rate = bau_surv_rate.rename(columns=rename_to)
        pop.loc[:, int_cols] = pop.loc[:, int_cols].mul(int_surv_rate)

        # Account for transitions between bins.
        # Note that the order of evaluation matters.
        suffixes = ["", "_intervention"]
        # First, accumulate the final post-exposure bin.
        if self.bin_years > 0:
            for suffix in suffixes:
                accum_col = "{}{}.{}".format(self.risk, suffix, self.bin_years + 1)
                from_col = "{}{}.{}".format(self.risk, suffix, self.bin_years)
                pop[accum_col] += pop[from_col]
        # Then increase time since exposure for all other post-exposure bins.
        for n_years in reversed(range(self.bin_years)):
            for suffix in suffixes:
                source_col = "{}{}.{}".format(self.risk, suffix, n_years)
                dest_col = "{}{}.{}".format(self.risk, suffix, n_years + 1)
                pop[dest_col] = pop[source_col]

        # Account for incidence and remission.
        col_no = "{}.no".format(self.risk)
        col_int_no = "{}_intervention.no".format(self.risk)
        col_yes = "{}.yes".format(self.risk)
        col_int_yes = "{}_intervention.yes".format(self.risk)
        col_zero = "{}.0".format(self.risk)
        col_int_zero = "{}_intervention.0".format(self.risk)

        inc = inc_rate * pop[col_no]
        int_inc = int_inc_rate * pop[col_int_no]
        rem = rem_rate * pop[col_yes]
        int_rem = int_rem_rate * pop[col_int_yes]

        # Account for the effects of a tobacco tax.
        if self.tobacco_tax:
            # The tax has a scaling effect (reduction) on incidence, and
            # causes additional remission.
            tax_inc = self.tax_effect_inc(idx)
            tax_rem = self.tax_effect_rem(idx)
            int_inc = int_inc * tax_inc
            int_rem = int_rem + (1 - tax_rem) * pop[col_int_yes]

        # Apply the incidence rate to the never-exposed population.
        pop[col_no] = pop[col_no] - inc
        pop[col_int_no] = pop[col_int_no] - int_inc
        # Incidence and remission affect who is currently exposed.
        pop[col_yes] = pop[col_yes] + inc - rem
        pop[col_int_yes] = pop[col_int_yes] + int_inc - int_rem
        # Those who have just remitted enter the first post-remission bin.
        pop[col_zero] = rem
        pop[col_int_zero] = int_rem

        self.population_view.update(pop)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def incidence_adjustment(
        self, disease: str, index: pd.Index, incidence_rate: pd.Series
    ) -> pd.Series:
        """Modify a disease incidence rate in the intervention scenario.

        Parameters
        ----------
        disease
            The name of the disease.
        index
            The index into the population life table.
        incidence_rate
            The un-adjusted disease incidence rate.

        """
        # Multiply the population in each bin by the associated relative risk.
        bin_cols = self.get_bin_names()
        pop = self.population_view.get(index)
        incidence_rr = self.dis_rr[disease](pop.index)[bin_cols]
        rr_values = pop[bin_cols] * incidence_rr

        # Calculate the mean relative-risk for the BAU scenario.
        bau_prefix = "{}.".format(self.risk)
        bau_cols = [c for c in bin_cols if c.startswith(bau_prefix)]
        # Sum over all of the bins in each row.
        mean_bau_rr = rr_values[bau_cols].sum(axis=1) / pop[bau_cols].sum(axis=1)
        # Handle cases where the population size is zero.
        mean_bau_rr = mean_bau_rr.fillna(1.0)

        # Calculate the mean relative-risk for the intervention scenario.
        int_prefix = "{}_intervention.".format(self.risk)
        int_cols = [c for c in bin_cols if c.startswith(int_prefix)]
        # Sum over all of the bins in each row.
        mean_int_rr = rr_values[int_cols].sum(axis=1) / pop[int_cols].sum(axis=1)
        # Handle cases where the population size is zero.
        mean_int_rr = mean_int_rr.fillna(1.0)

        # Calculate the disease incidence PIF for the intervention scenario.
        pif = (mean_bau_rr - mean_int_rr) / mean_bau_rr
        pif = pif.fillna(0.0)
        return incidence_rate * (1 - pif)


def pivot_load(builder: Builder, entity_key: str) -> pd.DataFrame:
    """Helper method for loading dataframe from artifact.

    Performs a long to wide conversion if dataframe has an index column
    named 'measure'.

    Parameters
    ----------
    builder
        The builder object for the simulation.
    entity_key
        The key for the entity to be loaded.

    Returns
    -------
    pd.DataFrame
        The loaded data and potentially pivoted data.
    """
    data = builder.data.load(entity_key)

    if "measure" in data.columns:
        data = (
            data.pivot_table(
                index=[i for i in data.columns if i not in ["measure", "value"]],
                columns="measure",
                values="value",
            )
            .rename_axis(None, axis=1)
            .reset_index()
        )

    return data
