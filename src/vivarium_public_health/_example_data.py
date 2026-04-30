"""
Example data for tutorials and interactive exploration.

This private module builds small, self-contained DataFrames that match the
column layout expected by :mod:`vivarium_public_health` components.
The data uses uniform rates and round population counts so that tutorial
output is easy to follow.  Age bins follow standard GBD definitions.

See the :doc:`/tutorials/population` and :doc:`/tutorials/disease` tutorials
for usage.
"""

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.artifact import ArtifactManager
from vivarium.framework.configuration import build_simulation_configuration

#############
# Constants #
#############

_YEAR_START = 1990
_YEAR_END = 2010
_LOCATION = "Kenya"

_YEAR_BINS: list[tuple[int, int]] = list(
    zip(range(_YEAR_START, _YEAR_END), range(_YEAR_START + 1, _YEAR_END + 1))
)


############
# Age bins #
############

_AGE_BIN_TUPLES: list[tuple[float, float, str]] = [
    (0.0, 0.01917808, "Early Neonatal"),
    (0.01917808, 0.07671233, "Late Neonatal"),
    (0.07671233, 1.0, "Post Neonatal"),
    (1.0, 5.0, "1 to 4"),
    (5.0, 10.0, "5 to 9"),
    (10.0, 15.0, "10 to 14"),
    (15.0, 20.0, "15 to 19"),
    (20.0, 25.0, "20 to 24"),
    (25.0, 30.0, "25 to 29"),
    (30.0, 35.0, "30 to 34"),
    (35.0, 40.0, "35 to 39"),
    (40.0, 45.0, "40 to 44"),
    (45.0, 50.0, "45 to 49"),
    (50.0, 55.0, "50 to 54"),
    (55.0, 60.0, "55 to 59"),
    (60.0, 65.0, "60 to 64"),
    (65.0, 70.0, "65 to 69"),
    (70.0, 75.0, "70 to 74"),
    (75.0, 80.0, "75 to 79"),
    (80.0, 85.0, "80 to 84"),
    (85.0, 90.0, "85 to 89"),
    (90.0, 95.0, "90 to 94"),
    (95.0, 125.0, "95 plus"),
]

_AGE_BINS: list[tuple[float, float]] = [(a, b) for a, b, _ in _AGE_BIN_TUPLES]


##############################################
# Data builders - one per artifact key shape #
##############################################


def _age_sex_year_grid() -> pd.DataFrame:
    """Return a DataFrame with every age x sex x year combination."""
    rows = list(product(_AGE_BINS, ("Male", "Female"), _YEAR_BINS))
    mins, maxes = zip(*[r[0] for r in rows])
    sex_col = [r[1] for r in rows]
    y_starts, y_ends = zip(*[r[2] for r in rows])
    return pd.DataFrame(
        {
            "age_start": mins,
            "age_end": maxes,
            "sex": sex_col,
            "year_start": y_starts,
            "year_end": y_ends,
        }
    )


def age_bins() -> pd.DataFrame:
    """Return the standard 23 GBD age-bin table.

    Columns: ``age_start``, ``age_end``, ``age_group_name``.
    """
    idx = pd.MultiIndex.from_tuples(
        _AGE_BIN_TUPLES,
        names=["age_start", "age_end", "age_group_name"],
    )
    return pd.DataFrame(index=idx).reset_index()


def population_structure() -> pd.DataFrame:
    """Return a uniform population structure across GBD age bins.

    Each demographic cell's ``value`` is proportional to the width of the
    age bin (100 people per year of age), making the numbers easy to reason
    about.

    Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
    ``year_end``, ``location``, ``value``.
    """
    df = _age_sex_year_grid()
    df["location"] = _LOCATION
    df["value"] = 100 * (df["age_end"] - df["age_start"])
    return df


def theoretical_minimum_risk_life_expectancy() -> pd.DataFrame:
    """Return a flat TMRLE of 98 years for every one-year age bin.

    Columns: ``age_start``, ``age_end``, ``value``.
    """
    ages = np.arange(0, 126)
    return pd.DataFrame(
        {
            "age_start": ages[:-1].astype(float),
            "age_end": ages[1:].astype(float),
            "value": 98.0,
        }
    )


def live_births_by_sex(annual_births_per_sex: float = 500.0) -> pd.DataFrame:
    """Return crude live-birth data for ``covariate.live_births_by_sex.estimate``.

    Each row gives the number of live births for one year × sex ×
    parameter combination.

    Parameters
    ----------
    annual_births_per_sex
        Number of live births per sex per year.

    Returns
    -------
    pandas.DataFrame
        Columns: ``year_start``, ``year_end``, ``sex``, ``parameter``, ``value``.
    """
    years = _YEAR_BINS
    sexes = ("Female", "Male")
    rows = list(product(years, sexes))
    y_starts, y_ends = zip(*[r[0] for r in rows])
    sex_col = [r[1] for r in rows]

    return pd.DataFrame(
        {
            "year_start": y_starts,
            "year_end": y_ends,
            "sex": sex_col,
            "parameter": "mean_value",
            "value": annual_births_per_sex,
        }
    )


def age_specific_fertility_rate(rate: float = 0.05) -> pd.DataFrame:
    """Return a uniform ASFR for ``covariate.age_specific_fertility_rate.estimate``.

    Parameters
    ----------
    rate
        The constant fertility rate applied to all age/year cells.

    Returns
    -------
    pandas.DataFrame
        Columns: ``year_start``, ``year_end``, ``age_start``, ``age_end``,
        ``sex``, ``parameter``, ``value``.
    """
    bins = _AGE_BINS
    years = _YEAR_BINS
    sexes = ("Female", "Male")
    parameters = ("mean_value", "lower_value", "upper_value")

    rows = list(product(years, bins, sexes, parameters))
    y_starts, y_ends = zip(*[r[0] for r in rows])
    mins, maxes = zip(*[r[1] for r in rows])
    sex_col = [r[2] for r in rows]
    param_col = [r[3] for r in rows]

    return pd.DataFrame(
        {
            "year_start": y_starts,
            "year_end": y_ends,
            "age_start": mins,
            "age_end": maxes,
            "sex": sex_col,
            "parameter": param_col,
            "value": rate,
        }
    )


########################
# Disease data helpers #
########################


def disease_prevalence(rate: float = 0.0) -> pd.DataFrame:
    """Return a uniform prevalence table for ``cause.{cause}.prevalence``.

    Parameters
    ----------
    rate
        The constant prevalence applied to all age/year/sex cells.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``value``.
    """
    return _build_cause_table(rate)


def disease_incidence_rate(rate: float = 0.001) -> pd.DataFrame:
    """Return a uniform incidence rate table for ``cause.{cause}.incidence_rate``.

    Parameters
    ----------
    rate
        The constant incidence rate applied to all age/year/sex cells.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``value``.
    """
    return _build_cause_table(rate)


def disease_remission_rate(rate: float = 0.0) -> pd.DataFrame:
    """Return a uniform remission rate table for ``cause.{cause}.remission_rate``.

    Parameters
    ----------
    rate
        The constant remission rate applied to all age/year/sex cells.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``value``.
    """
    return _build_cause_table(rate)


def disease_excess_mortality_rate(rate: float = 0.0) -> pd.DataFrame:
    """Return a uniform excess mortality rate table for ``cause.{cause}.excess_mortality_rate``.

    Parameters
    ----------
    rate
        The constant excess mortality rate applied to all age/year/sex cells.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``value``.
    """
    return _build_cause_table(rate)


def disease_cause_specific_mortality_rate(rate: float = 0.0) -> pd.DataFrame:
    """Return a uniform CSMR table for ``cause.{cause}.cause_specific_mortality_rate``.

    Parameters
    ----------
    rate
        The constant cause-specific mortality rate.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``value``.
    """
    return _build_cause_table(rate)


def disease_disability_weight(weight: float = 0.0) -> pd.DataFrame:
    """Return a disability weight table for ``cause.{cause}.disability_weight``.

    Parameters
    ----------
    weight
        The constant disability weight.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with a ``value`` column.
    """
    return pd.DataFrame({"value": [weight]})


def disease_restrictions(yld_only: bool = False) -> dict:
    """Return a restrictions dict for ``cause.{cause}.restrictions``.

    Parameters
    ----------
    yld_only
        Whether the cause only contributes to years lived with disability
        (no mortality).

    Returns
    -------
    dict
        A dict with key ``yld_only``.
    """
    return {"yld_only": yld_only}


def _build_cause_table(value: float) -> pd.DataFrame:
    """Build an age x sex x year table with a constant value.

    This is the standard shape for most cause-level measures (prevalence,
    incidence, remission, excess mortality, CSMR).
    """
    df = _age_sex_year_grid()
    df["value"] = value
    return df


############################
# Example artifact manager #
############################

_ARTIFACT_DATA: dict[str, object] = {
    "population.structure": population_structure,
    "population.age_bins": age_bins,
    "population.location": lambda: _LOCATION,
    "population.theoretical_minimum_risk_life_expectancy": theoretical_minimum_risk_life_expectancy,
    "population.demographic_dimensions": lambda: population_structure().drop(
        columns=["location", "value"]
    ),
    "covariate.live_births_by_sex.estimate": live_births_by_sex,
    "covariate.age_specific_fertility_rate.estimate": age_specific_fertility_rate,
    # Mortality data - zero by default so simulants stay alive in examples.
    "cause.all_causes.cause_specific_mortality_rate": lambda: 0.0,
    # Tutorial-specific cause data.  Rates are high enough to guarantee visible
    # state transitions within 5-10 time steps (each ~30.5 days).
    "cause.test_cause.incidence_rate": lambda: disease_incidence_rate(rate=0.5),
    # Remission of 5.0/person-year ensures rapid recovery for SIS/SIR demos.
    "cause.test_cause.remission_rate": lambda: disease_remission_rate(rate=5.0),
    "cause.neonatal_cause.incidence_rate": lambda: disease_incidence_rate(rate=0.5),
    # Birth prevalence of 5% is high enough to see neonatal cases at birth.
    "cause.neonatal_cause.birth_prevalence": lambda: disease_prevalence(rate=0.05),
    "cause.diarrheal_diseases.incidence_rate": lambda: disease_incidence_rate(rate=0.5),
    # Remission of 1.0/person-year balances infected/susceptible pools for demos.
    "cause.diarrheal_diseases.remission_rate": lambda: disease_remission_rate(rate=1.0),
}


# Default disease data keyed by measure name.  _ExampleArtifact uses these
# as fallbacks for any ``cause.{name}.{measure}`` key not in _ARTIFACT_DATA.
_CAUSE_DEFAULTS: dict[str, Callable[[], Any]] = {
    "prevalence": disease_prevalence,
    "birth_prevalence": disease_prevalence,
    "cause_specific_mortality_rate": disease_cause_specific_mortality_rate,
    "excess_mortality_rate": disease_excess_mortality_rate,
    "remission_rate": disease_remission_rate,
    "incidence_rate": disease_incidence_rate,
    "disability_weight": disease_disability_weight,
    "restrictions": disease_restrictions,
}


class _ExampleArtifact:
    """Serve pre-built example DataFrames by artifact key."""

    def __init__(self) -> None:
        self._overrides: dict[str, object] = {}

    def load(self, entity_key: str):
        if entity_key in self._overrides:
            return self._overrides[entity_key]
        if entity_key in _ARTIFACT_DATA:
            value = _ARTIFACT_DATA[entity_key]
            return value() if callable(value) else value
        # Fall back to default disease data for cause.{name}.{measure} keys.
        parts = entity_key.split(".")
        if len(parts) == 3 and parts[0] == "cause":
            measure = parts[2]
            if measure in _CAUSE_DEFAULTS:
                value = _CAUSE_DEFAULTS[measure]
                return value() if callable(value) else value
        raise KeyError(f"No example data for artifact key {entity_key!r}")

    def write(self, entity_key: str, data: object) -> None:
        self._overrides[entity_key] = data


class ExampleArtifactManager(ArtifactManager):
    """Artifact manager that serves example data without requiring an HDF file.

    Use this in a plugin configuration so that tutorials and interactive
    sessions can run without access to real GBD data::

        base_plugins = LayeredConfigTree({
            "required": {
                "data": {
                    "controller": "vivarium_public_health._example_data.ExampleArtifactManager",
                    "builder_interface": "vivarium.framework.artifact.ArtifactInterface",
                }
            }
        })
    """

    def __init__(self) -> None:
        super().__init__()
        self.artifact = self._load_artifact(None)

    @property
    def name(self) -> str:
        return "example_artifact_manager"

    def setup(self, builder) -> None:
        pass

    def load(self, entity_key: str, *args, **kwargs):
        return self.artifact.load(entity_key)

    def write(self, entity_key: str, data: object) -> None:
        self.artifact.write(entity_key, data)

    def _load_artifact(self, _) -> _ExampleArtifact:
        return _ExampleArtifact()


####################
# Tutorial helpers #
####################

#: Plugin configuration that wires up the example artifact manager.
BASE_PLUGINS = LayeredConfigTree(
    {
        "required": {
            "data": {
                "controller": "vivarium_public_health._example_data.ExampleArtifactManager",
                "builder_interface": "vivarium.framework.artifact.ArtifactInterface",
            }
        }
    }
)


def make_base_config() -> LayeredConfigTree:
    """Return a fresh base configuration for tutorial examples."""
    config = build_simulation_configuration()
    config.update(
        {
            "time": {
                "start": {"year": _YEAR_START},
                "end": {"year": _YEAR_END},
                "step_size": 30.5,
            },
            "randomness": {"key_columns": ["entrance_time", "age"]},
        },
        layer="model_override",
    )
    return config
