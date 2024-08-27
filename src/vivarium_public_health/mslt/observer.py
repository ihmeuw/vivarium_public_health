"""
=========
Observers
=========

This module contains tools for recording various outputs of interest in
multi-state lifetable simulations.

"""

from typing import List, Optional

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


def output_file(config, suffix, sep="_", ext="csv") -> str:
    """Determine the output file name for an observer, based on the prefix
    defined in ``config.observer.output_prefix`` and the (optional)
    ``config.input_data.input_draw_number``.

    Parameters
    ----------
    config
        The builder configuration object.
    suffix
        The observer-specific suffix.
    sep
        The separator between prefix, suffix, and draw number.
    ext
        The output file extension.

    Returns
    -------
        The output file name for the observer.
    """
    if "observer" not in config:
        raise ValueError("observer.output_prefix not defined")
    if "output_prefix" not in config.observer:
        raise ValueError("observer.output_prefix not defined")
    prefix = config.observer.output_prefix
    if "input_draw_number" in config.input_data:
        draw = config.input_data.input_draw_number
    else:
        draw = 0
    out_file = prefix + sep + suffix
    if draw > 0:
        out_file += "{}{}".format(sep, draw)
    out_file += ".{}".format(ext)
    return out_file


class MorbidityMortality(Component):
    """This class records the all-cause morbidity and mortality rates for each
    cohort at each year of the simulation.

    Attributes
    ----------
    output_suffix
        The suffix for the CSV file in which to record the
        morbidity and mortality data.
    clock
        The simulation clock.
    tables
        The tables of morbidity and mortality data.
    table_cols
        The columns in the tables.
    output_file
        The output file for the morbidity and mortality data.
    """

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [
            "age",
            "sex",
            "population",
            "bau_population",
            "acmr",
            "bau_acmr",
            "pr_death",
            "bau_pr_death",
            "deaths",
            "bau_deaths",
            "yld_rate",
            "bau_yld_rate",
            "person_years",
            "bau_person_years",
            "HALY",
            "bau_HALY",
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, output_suffix: str = "mm"):
        super().__init__()
        self.output_suffix = output_suffix

    def setup(self, builder: Builder) -> None:
        # Record the key columns from the core multi-state life table.
        self.clock = builder.time.clock()

        self.tables = []
        self.table_cols = self.columns_required + [
            "year",
            "prev_population",
            "bau_prev_population",
        ]

        self.output_file = output_file(builder.configuration, self.output_suffix)

    ########################
    # Event-driven methods #
    ########################

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        if len(pop.index) == 0:
            # No tracked population remains.
            return

        pop["year"] = self.clock().year
        # Record the population size prior to the deaths.
        pop["prev_population"] = pop["population"] + pop["deaths"]
        pop["bau_prev_population"] = pop["bau_population"] + pop["bau_deaths"]
        self.tables.append(pop[self.table_cols])

    def on_simulation_end(self, event: Event) -> None:
        data = pd.concat(self.tables, ignore_index=True)
        data["year_of_birth"] = data["year"] - data["age"]
        # Sort the table by cohort (i.e., generation and sex), and then by
        # calendar year, so that results are output in the same order as in
        # the spreadsheet models.
        data = data.sort_values(by=["year_of_birth", "sex", "age"], axis=0)
        data = data.reset_index(drop=True)
        # Re-order the table columns.
        cols = ["year_of_birth"] + self.table_cols
        data = data[cols]
        # Calculate life expectancy and HALE for the BAU and intervention,
        # with respect to the initial population, not the survivors.
        data["LE"] = self.calculate_LE(data, "person_years", "prev_population")
        data["bau_LE"] = self.calculate_LE(data, "bau_person_years", "bau_prev_population")
        data["HALE"] = self.calculate_LE(data, "HALY", "prev_population")
        data["bau_HALE"] = self.calculate_LE(data, "bau_HALY", "bau_prev_population")
        data.to_csv(self.output_file, index=False)

    ##################
    # Helper methods #
    ##################

    def calculate_LE(self, table, py_col, denom_col) -> pd.Series:
        """Calculate the life expectancy for each cohort at each time-step.

        Parameters
        ----------
        table
            The population life table.
        py_col
            The name of the person-years column.
        denom_col
            The name of the population denominator column.

        Returns
        -------
            The life expectancy for each table row.
        """
        # Group the person-years by cohort.
        group_cols = ["year_of_birth", "sex"]
        subset_cols = group_cols + [py_col]
        grouped = table.loc[:, subset_cols].groupby(by=group_cols)[py_col]
        # Calculate the reverse-cumulative sums of the adjusted person-years
        # (i.e., the present and future person-years) by:
        #   (a) reversing the adjusted person-years values in each cohort;
        #   (b) calculating the cumulative sums in each cohort; and
        #   (c) restoring the original order.
        cumsum = grouped.apply(lambda x: pd.Series(x[::-1].cumsum()).iloc[::-1])
        return cumsum / table[denom_col]


class Disease(Component):
    """This class records the disease incidence rate and disease prevalence for
    each cohort at each year of the simulation.

    Attributes
    ----------
    disease
        The name of the chronic disease.
    output_suffix
        The suffix for the CSV file in which to record the
        disease data.
    bau_S_col
        The name of the BAU susceptible column.
    bau_C_col
        The name of the BAU chronic column.
    int_S_col
        The name of the intervention susceptible column.
    int_C_col
        The name of the intervention chronic column.
    bau_incidence
        The incidence rate for the BAU scenario.
    int_incidence
        The incidence rate for the intervention scenario.
    tables
        The tables of disease data.
    table_cols
        The columns in the tables.
    clock
        The simulation clock.
    output_file
        The output file for the disease data.

    """

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> Optional[List[str]]:
        return [
            "age",
            "sex",
            self.bau_S_col,
            self.bau_C_col,
            self.int_S_col,
            self.int_C_col,
        ]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, disease: str, output_suffix: Optional[str] = None):
        super().__init__()
        self.disease = disease
        if output_suffix is None:
            output_suffix = disease.lower()
        self.output_suffix = output_suffix

        self.bau_S_col = "{}_S".format(self.disease)
        self.bau_C_col = "{}_C".format(self.disease)
        self.int_S_col = "{}_S_intervention".format(self.disease)
        self.int_C_col = "{}_C_intervention".format(self.disease)

    def setup(self, builder: Builder) -> None:
        bau_incidence_value = "{}.incidence".format(self.disease)
        int_incidence_value = "{}_intervention.incidence".format(self.disease)
        self.bau_incidence = builder.value.get_value(bau_incidence_value)
        self.int_incidence = builder.value.get_value(int_incidence_value)

        self.tables = []
        self.table_cols = [
            "sex",
            "age",
            "year",
            "bau_incidence",
            "int_incidence",
            "bau_prevalence",
            "int_prevalence",
            "bau_deaths",
            "int_deaths",
        ]
        self.clock = builder.time.clock()
        self.output_file = output_file(builder.configuration, self.output_suffix)

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        if len(pop.index) == 0:
            # No tracked population remains.
            return

        pop["year"] = self.clock().year
        pop["bau_incidence"] = self.bau_incidence(event.index)
        pop["int_incidence"] = self.int_incidence(event.index)
        pop["bau_prevalence"] = pop[self.bau_C_col] / (
            pop[self.bau_C_col] + pop[self.bau_S_col]
        )
        pop["int_prevalence"] = pop[self.int_C_col] / (
            pop[self.bau_C_col] + pop[self.bau_S_col]
        )
        pop["bau_deaths"] = 1000 - pop[self.bau_S_col] - pop[self.bau_C_col]
        pop["int_deaths"] = 1000 - pop[self.int_S_col] - pop[self.int_C_col]
        self.tables.append(pop.loc[:, self.table_cols])

    def on_simulation_end(self, event: Event) -> None:
        data = pd.concat(self.tables, ignore_index=True)
        data["diff_incidence"] = data["int_incidence"] - data["bau_incidence"]
        data["diff_prevalence"] = data["int_prevalence"] - data["bau_prevalence"]
        data["year_of_birth"] = data["year"] - data["age"]
        data["disease"] = self.disease
        # Sort the table by cohort (i.e., generation and sex), and then by
        # calendar year, so that results are output in the same order as in
        # the spreadsheet models.
        data = data.sort_values(by=["year_of_birth", "sex", "age"], axis=0)
        data = data.reset_index(drop=True)
        # Re-order the table columns.
        diff_cols = ["diff_incidence", "diff_prevalence"]
        cols = ["disease", "year_of_birth"] + self.table_cols + diff_cols
        data = data[cols]
        data.to_csv(self.output_file, index=False)


class TobaccoPrevalence(Component):
    """This class records the prevalence of tobacco use in the population.

    Attributes
    ----------
    output_suffix
        The suffix for the CSV file in which to record the
        prevalence data.
    config
        The builder configuration object.
    clock
        The simulation clock.
    bin_years
        The number of years post-exposure to consider.
    tables
        The tables of tobacco prevalence data.
    table_cols
        The columns in the tables.
    output_file
        The output file for the tobacco prevalence data.
    """

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", "sex", "bau_population", "population"] + self._bin_names

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, output_suffix: str = "tobacco"):
        super().__init__()
        self.output_suffix = output_suffix
        self._bin_names = []

    def setup(self, builder: Builder) -> None:
        self._bin_names = self.get_bin_names()

        self.config = builder.configuration
        self.clock = builder.time.clock()
        self.bin_years = int(self.config["tobacco"]["delay"])

        self.tables = []
        self.table_cols = [
            "age",
            "sex",
            "year",
            "bau_no",
            "bau_yes",
            "bau_previously",
            "bau_population",
            "int_no",
            "int_yes",
            "int_previously",
            "int_population",
        ]

        self.output_file = output_file(builder.configuration, self.output_suffix)

    #################
    # Setup methods #
    #################

    def get_bin_names(self) -> list[str]:
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

        Returns
        -------
            The bin names for tobacco use.
        """
        if self.bin_years == 0:
            delay_bins = [str(0)]
        else:
            delay_bins = [str(s) for s in range(self.bin_years + 2)]
        bins = ["no", "yes"] + delay_bins
        bau_bins = ["{}.{}".format("tobacco", bin) for bin in bins]
        int_bins = ["{}_intervention.{}".format("tobacco", bin) for bin in bins]
        all_bins = bau_bins + int_bins
        return all_bins

    ########################
    # Event-driven methods #
    ########################

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        if len(pop.index) == 0:
            # No tracked population remains.
            return

        bau_cols = [c for c in pop.columns.values if c.startswith("{}.".format("tobacco"))]
        int_cols = [
            c
            for c in pop.columns.values
            if c.startswith("{}_intervention.".format("tobacco"))
        ]

        bau_denom = pop.reindex(columns=bau_cols).sum(axis=1)
        int_denom = pop.reindex(columns=int_cols).sum(axis=1)

        # Normalise prevalence with respect to the total population.
        pop["bau_no"] = pop["{}.no".format("tobacco")] / bau_denom
        pop["bau_yes"] = pop["{}.yes".format("tobacco")] / bau_denom
        pop["bau_previously"] = 1 - pop["bau_no"] - pop["bau_yes"]
        pop["int_no"] = pop["{}_intervention.no".format("tobacco")] / int_denom
        pop["int_yes"] = pop["{}_intervention.yes".format("tobacco")] / int_denom
        pop["int_previously"] = 1 - pop["int_no"] - pop["int_yes"]

        pop = pop.rename(columns={"population": "int_population"})

        pop["year"] = self.clock().year
        self.tables.append(pop.reindex(columns=self.table_cols).reset_index(drop=True))

    def on_simulation_end(self, event: Event) -> None:
        data = pd.concat(self.tables, ignore_index=True)
        data["year_of_birth"] = data["year"] - data["age"]
        # Sort the table by cohort (i.e., generation and sex), and then by
        # calendar year, so that results are output in the same order as in
        # the spreadsheet models.
        data = data.sort_values(by=["year_of_birth", "sex", "age"], axis=0)
        data = data.reset_index(drop=True)
        # Re-order the table columns.
        cols = ["year_of_birth"] + self.table_cols
        data = data.reindex(columns=cols)
        data.to_csv(self.output_file, index=False)
