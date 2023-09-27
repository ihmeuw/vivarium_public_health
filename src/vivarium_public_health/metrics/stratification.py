"""
==================
Results Stratifier
==================

This module contains tools for stratifying observed quantities
by specified characteristics through the vivarium results interface.
"""

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder


class ResultsStratifier(Component):
    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.age_bins = self.get_age_bins(builder)
        self.start_year = builder.configuration.time.start.year
        self.end_year = builder.configuration.time.end.year

        self.register_stratifications(builder)

    #################
    # Setup methods #
    #################

    def register_stratifications(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "age_group",
            self.age_bins["age_group_name"].to_list(),
            self.map_age_groups,
            is_vectorized=True,
            requires_columns=["age"],
        )
        builder.results.register_stratification(
            "current_year",
            [str(year) for year in range(self.start_year, self.end_year + 1)],
            self.map_year,
            is_vectorized=True,
            requires_columns=["current_time"],
        )
        # TODO [MIC-4232]: simulants occasionally have event year of end_year_year+1 if the end time plus step size
        #  lands in the next year. possible solution detailed in ticket
        # builder.results.register_stratification(
        #     "event_year",
        #     [str(year) for year in range(self.start_year, self.end_year + 1)],
        #     self.map_year,
        #     is_vectorized=True,
        #     requires_columns=["event_time"],
        # )
        # TODO [MIC-3892]: simulants occasionally have entrance year of start_year-1 if the start time minus step size
        #  lands in the previous year. possible solution detailed in ticket
        # builder.results.register_stratification(
        #     "entrance_year",
        #     [str(year) for year in range(self.start_year, self.end_year + 1)],
        #     self.map_year,
        #     is_vectorized=True,
        #     requires_columns=["entrance_time"],
        # )
        # TODO [MIC-4083]: Known bug with this registration
        # builder.results.register_stratification(
        #     "exit_year",
        #     [str(year) for year in range(self.start_year, self.end_year + 1)] + ["nan"],
        #     self.map_year,
        #     is_vectorized=True,
        #     requires_columns=["exit_time"],
        # )
        builder.results.register_stratification(
            "sex", ["Female", "Male"], requires_columns=["sex"]
        )

    ###########
    # Mappers #
    ###########

    def map_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        """Map age with age group name strings

        Parameters
        ----------
        pop
            A DataFrame with one column, an age to be mapped to an age group name string

        Returns
        ------
        pandas.Series
            A pd.Series with age group name string corresponding to the pop passed into the function
        """
        bins = self.age_bins["age_start"].to_list() + [self.age_bins["age_end"].iloc[-1]]
        labels = self.age_bins["age_group_name"].to_list()
        age_group = pd.cut(pop.squeeze(axis=1), bins, labels=labels).rename("age_group")
        return age_group

    @staticmethod
    def map_year(pop: pd.DataFrame) -> pd.Series:
        """Map datetime with year

        Parameters
        ----------
        pop
            A DataFrame with one column, a datetime to be mapped to year

        Returns
        ------
        pandas.Series
            A pd.Series with years corresponding to the pop passed into the function
        """
        return pop.squeeze(axis=1).dt.year.apply(str)

    @staticmethod
    def get_age_bins(builder: Builder) -> pd.DataFrame:
        raw_age_bins = builder.data.load("population.age_bins")
        age_start = builder.configuration.population.age_start
        exit_age = builder.configuration.population.exit_age

        age_start_mask = age_start < raw_age_bins["age_end"]
        exit_age_mask = raw_age_bins["age_start"] < exit_age if exit_age else True

        age_bins = raw_age_bins.loc[age_start_mask & exit_age_mask, :].copy()
        age_bins["age_group_name"] = (
            age_bins["age_group_name"].str.replace(" ", "_").str.lower()
        )
        return age_bins
