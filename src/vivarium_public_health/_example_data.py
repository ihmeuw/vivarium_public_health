"""
Example data for tutorials and interactive exploration.

This private module builds small, self-contained DataFrames that match the
column layout expected by :mod:`vivarium_public_health.population` components.
The data uses uniform rates and round population counts so that tutorial
output is easy to follow.  Age bins follow standard GBD definitions.

See the :doc:`/tutorials/population` tutorial for usage.
"""

from itertools import product

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
# Data builders — one per artifact key shape #
##############################################


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
    bins = _AGE_BINS
    sexes = ("Male", "Female")
    years = _YEAR_BINS

    rows = list(product(bins, sexes, years))
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
            "location": _LOCATION,
            "value": 100 * (np.array(maxes) - np.array(mins)),
        }
    )


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
    # Mortality data — zero by default so simulants stay alive in examples.
    "cause.all_causes.cause_specific_mortality_rate": lambda: 0.0,
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
