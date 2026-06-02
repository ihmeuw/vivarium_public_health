"""
Example data for tutorials and interactive exploration.

This private module builds small, self-contained DataFrames that match the
column layout expected by :mod:`vivarium_public_health` components.
The data uses uniform rates and round population counts so that tutorial
output is easy to follow.  Age bins follow standard GBD definitions.

See the :doc:`/tutorials/population`, :doc:`/tutorials/disease`, and
:doc:`/tutorials/risk` tutorials for usage.
"""

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.artifact import ArtifactManager
from vivarium.framework.configuration import build_simulation_configuration
from vivarium.framework.engine import Builder

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


def get_age_sex_year_grid() -> pd.DataFrame:
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
    df = get_age_sex_year_grid()
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


def build_cause_table(value: float = 0.0) -> pd.DataFrame:
    """Return an age x sex x year table with a constant value column.

    This is the standard shape for most cause-level measures (prevalence,
    incidence, remission, excess mortality, CSMR).

    Parameters
    ----------
    value
        The constant value applied to all age/year/sex cells.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``value``.
    """
    df = get_age_sex_year_grid()
    df["value"] = value
    return df


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


###########################
# Risk factor data helpers #
###########################


def _build_dichotomous_table(
    exposed_value: float,
    unexposed_value: float,
    extra_columns: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build a dichotomous (exposed/unexposed) table on the age x sex x year grid."""
    df = get_age_sex_year_grid()
    exposed = df.copy()
    exposed["parameter"] = "exposed"
    exposed["value"] = exposed_value
    unexposed = df.copy()
    unexposed["parameter"] = "unexposed"
    unexposed["value"] = unexposed_value
    result = pd.concat([exposed, unexposed], ignore_index=True)
    if extra_columns:
        for col, val in extra_columns.items():
            result[col] = val
    return result


def risk_exposure_dichotomous(proportion_exposed: float = 0.6) -> pd.DataFrame:
    """Return dichotomous exposure data for ``risk_factor.{name}.exposure``.

    Parameters
    ----------
    proportion_exposed
        The fraction of the population in the exposed category.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``parameter``, ``value``.
    """
    return _build_dichotomous_table(proportion_exposed, 1 - proportion_exposed)


def risk_relative_risk_dichotomous(
    rr_exposed: float = 2.0,
    target_entity: str = "test_cause",
    target_measure: str = "incidence_rate",
) -> pd.DataFrame:
    """Return dichotomous relative risk data for ``risk_factor.{name}.relative_risk``.

    Parameters
    ----------
    rr_exposed
        The relative risk for the exposed category.  The unexposed
        category always has a relative risk of 1.
    target_entity
        The name of the affected entity (e.g., a cause name).
    target_measure
        The affected measure (e.g., ``"incidence_rate"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``affected_entity``, ``affected_measure``,
        ``parameter``, ``value``.
    """
    return _build_dichotomous_table(
        exposed_value=rr_exposed,
        unexposed_value=1.0,
        extra_columns={
            "affected_entity": target_entity,
            "affected_measure": target_measure,
        },
    )


def risk_relative_risk_continuous(
    exposure_min: float = 1.0,
    exposure_max: float = 9.0,
    rr_min: float = 1.0,
    rr_max: float = 5.0,
    n_thresholds: int = 1000,
    target_entity: str = "test_cause",
    target_measure: str = "incidence_rate",
) -> pd.DataFrame:
    """Return continuous relative risk data for ``NonLogLinearRiskEffect``.

    Builds a DataFrame with ``n_thresholds`` evenly spaced exposure
    thresholds and linearly increasing RR values, suitable for use with
    :class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect`.

    Parameters
    ----------
    exposure_min
        Lower bound of the exposure range.
    exposure_max
        Upper bound of the exposure range.
    rr_min
        Relative risk at ``exposure_min``.
    rr_max
        Relative risk at ``exposure_max``.
    n_thresholds
        Number of exposure thresholds.
    target_entity
        The name of the affected entity.
    target_measure
        The affected measure.

    Returns
    -------
    pandas.DataFrame
        Columns: ``parameter``, ``value``, ``affected_entity``,
        ``affected_measure``, ``year_start``, ``year_end``.
    """
    thresholds = np.linspace(exposure_min, exposure_max, n_thresholds)
    rr_values = np.linspace(rr_min, rr_max, n_thresholds)
    return pd.DataFrame(
        {
            "parameter": thresholds,
            "value": rr_values,
            "affected_entity": target_entity,
            "affected_measure": target_measure,
            "year_start": _YEAR_START,
            "year_end": _YEAR_START + 1,
        }
    )


def risk_paf(
    paf_value: float = 0.0,
    target_entity: str = "test_cause",
    target_measure: str = "incidence_rate",
) -> pd.DataFrame:
    """Return PAF data for ``risk_factor.{name}.population_attributable_fraction``.

    Parameters
    ----------
    paf_value
        The population attributable fraction value.
    target_entity
        The name of the affected entity.
    target_measure
        The affected measure.

    Returns
    -------
    pandas.DataFrame
        Columns: ``age_start``, ``age_end``, ``sex``, ``year_start``,
        ``year_end``, ``affected_entity``, ``affected_measure``, ``value``.
    """
    df = get_age_sex_year_grid()
    df["affected_entity"] = target_entity
    df["affected_measure"] = target_measure
    df["value"] = paf_value
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
    "cause.test_cause.incidence_rate": lambda: build_cause_table(0.5),
    # Remission of 5.0/person-year ensures rapid recovery for SIS/SIR demos.
    "cause.test_cause.remission_rate": lambda: build_cause_table(5.0),
    "cause.neonatal_cause.incidence_rate": lambda: build_cause_table(0.5),
    # Birth prevalence of 5% is high enough to see neonatal cases at birth.
    "cause.neonatal_cause.birth_prevalence": lambda: build_cause_table(0.05),
    "cause.diarrheal_diseases.incidence_rate": lambda: build_cause_table(0.5),
    # Remission of 1.0/person-year balances infected/susceptible pools for demos.
    "cause.diarrheal_diseases.remission_rate": lambda: build_cause_table(1.0),
    # Risk factor data - dichotomous distribution with 60% exposed by default.
    "risk_factor.test_risk.distribution": lambda: "dichotomous",
    "risk_factor.test_risk.exposure": risk_exposure_dichotomous,
    "risk_factor.test_risk.relative_risk": risk_relative_risk_dichotomous,
    "risk_factor.test_risk.population_attributable_fraction": lambda: risk_paf(),
}


# Default disease data keyed by measure name.  _ExampleArtifact uses these
# as fallbacks for any ``cause.{name}.{measure}`` key not in _ARTIFACT_DATA.
_CAUSE_DEFAULTS: dict[str, Callable[[], Any]] = {
    "prevalence": build_cause_table,
    "birth_prevalence": build_cause_table,
    "cause_specific_mortality_rate": build_cause_table,
    "excess_mortality_rate": build_cause_table,
    "remission_rate": build_cause_table,
    "incidence_rate": build_cause_table,
    "disability_weight": disease_disability_weight,
    "restrictions": disease_restrictions,
}

# Default risk factor data keyed by measure name.  _ExampleArtifact uses these
# as fallbacks for any ``risk_factor.{name}.{measure}`` key not in _ARTIFACT_DATA.
# NOTE: The "relative_risk" fallback always targets "test_cause" because the
# callable is entity-unaware.  For other targets, register explicit artifact data
# or use scalar/DataFrame overrides via data_sources configuration.
_RISK_DEFAULTS: dict[str, Callable[[], Any]] = {
    "distribution": lambda: "dichotomous",
    "categories": lambda: {"exposed": "exposed", "unexposed": "unexposed"},
    "exposure": risk_exposure_dichotomous,
    "relative_risk": risk_relative_risk_dichotomous,
    "population_attributable_fraction": risk_paf,
    "tmred": lambda: {"distribution": "uniform", "min": 0, "max": 0, "inverted": False},
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
        # Fall back to default risk data for risk_factor.{name}.{measure} keys.
        # NOTE: Only the "risk_factor" prefix is supported here;
        # "alternative_risk_factor" is not currently handled.
        if len(parts) == 3 and parts[0] == "risk_factor":
            measure = parts[2]
            if measure in _RISK_DEFAULTS:
                value = _RISK_DEFAULTS[measure]
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


class ConstantRatePipeline(Component):
    """Register a constant-valued attribute pipeline for tutorial demonstrations.

    This component creates a simple attribute pipeline with
    ``replace_combiner`` (the default) that returns a fixed value for all
    simulants. It is useful for demonstrating pipeline modifiers like
    :class:`~vivarium_public_health.treatment.magic_wand.AbsoluteShift`
    without requiring a full disease model.

    Parameters
    ----------
    pipeline_name
        The name of the attribute pipeline to register
        (e.g. ``"test_cause.incidence_rate"``).
    rate
        The constant value the pipeline returns for every simulant.
    """

    def __init__(self, pipeline_name: str, rate: float = 0.1) -> None:
        super().__init__()
        self._pipeline_name = pipeline_name
        self._rate = rate

    @property
    def name(self) -> str:
        """The name of this component."""
        return f"constant_rate_pipeline.{self._pipeline_name}"

    def setup(self, builder: Builder) -> None:
        """Register the constant-valued attribute pipeline.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        builder.value.register_attribute_producer(
            self._pipeline_name,
            source=self._source,
        )

    def _source(self, index: pd.Index) -> pd.Series:
        """Return the constant rate for all simulants."""
        return pd.Series(self._rate, index=index)
