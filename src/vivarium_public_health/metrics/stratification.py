"""
==================
Results Stratifier
==================

This module contains tools for stratifying observed quantities
by specified characteristics through the vivarium results interface.
"""

import pandas as pd
from vivarium.framework.engine import Builder


class ResultsStratifier:
    name = "results_stratifier"

    configuration_defaults = {
        "stratification": {
            "default": [],
        }
    }

    def setup(self, builder: Builder):
        self.config = builder.configuration.stratification
        self.age_bins = self.get_age_bins(builder)

        self.start_year = builder.configuration.time.start.year
        self.end_year = builder.configuration.time.end.year

        builder.results.set_default_stratifications(self.config.stratification.default)
        self.register_stratifications(builder)

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
        builder.results.register_stratification(
            "event_year",
            [str(year) for year in range(self.start_year, self.end_year + 1)],
            self.map_year,
            is_vectorized=True,
            requires_columns=["event_time"],
        )
        builder.results.register_stratification(
            "entrance_year",
            [str(year) for year in range(self.start_year, self.end_year + 1)],
            self.map_year,
            is_vectorized=True,
            requires_columns=["entrance_time"],
        )
        builder.results.register_stratification(
            "exit_year",
            [str(year) for year in range(self.start_year, self.end_year + 1)],
            self.map_year,
            is_vectorized=True,
            requires_columns=["exit_time"],
        )

    def map_age_groups(self, pop: pd.DataFrame) -> pd.Series:
        """Map age with age group name strings

        Parameters
        ----------
        pop
            A DataFrame with one column, an age to be mapped to an age group name string

        Returns
        ------
        pd.Series
            A pd.Series with age group name string corresponding to the pop passed into the function
        """
        bins = self.age_bins.age_start.to_list() + [self.age_bins.age_end.iloc[-1]]
        labels = self.age_bins["age_group_name"].to_list()
        age_group = pd.cut(pop["age"], bins, labels=labels).rename("age_group")
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
        pd.Series
            A pd.Series with years corresponding to the pop passed into the function
        """
        return pop.squeeze(axis=1).dt.year

    @staticmethod
    def get_age_bins(builder: Builder):
        raw_age_bins = builder.data.load("population.age_bins")
        age_start = builder.configuration.population.age_start
        exit_age = builder.configuration.population.exit_age

        age_start_mask = age_start < raw_age_bins["age_end"]
        exit_age_mask = raw_age_bins["age_start"] < exit_age if exit_age else True

        age_bins = raw_age_bins.loc[age_start_mask & exit_age_mask, :]
        age_bins["age_group_name"] = (
            age_bins["age_group_name"].str.replace(" ", "_").str.lower()
        )
        return age_bins
