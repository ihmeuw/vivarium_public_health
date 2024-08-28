"""
=========================
Risk Data Transformations
=========================

This module contains tools for handling raw risk exposure and relative
risk data and performing any necessary data transformations.

"""

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_public_health.utilities import EntityString, TargetString

#############
# Utilities #
#############


def pivot_categorical(
    builder: Builder,
    risk: EntityString,
    data: pd.DataFrame,
    pivot_column: str = "parameter",
    reset_index: bool = True,
) -> pd.DataFrame:
    """Pivots data that is long on categories to be wide."""
    # todo remove dependency on artifact manager having exactly one value column
    value_column = builder.data.value_columns()(f"{risk}.exposure")[0]
    index_cols = [
        column for column in data.columns if column not in [value_column, pivot_column]
    ]
    data = data.pivot_table(index=index_cols, columns=pivot_column, values=value_column)
    if reset_index:
        data = data.reset_index()
    data.columns.name = None

    return data


##########################
# Exposure data handlers #
##########################


def get_exposure_post_processor(builder, risk: str):
    thresholds = builder.configuration[risk]["category_thresholds"]

    if thresholds:
        thresholds = [-np.inf] + thresholds + [np.inf]
        categories = [f"cat{i}" for i in range(1, len(thresholds))]

        def post_processor(exposure, _):
            return pd.Series(
                pd.cut(exposure, thresholds, labels=categories), index=exposure.index
            ).astype(str)

    else:
        post_processor = None

    return post_processor


def load_exposure_data(builder: Builder, risk: EntityString) -> pd.DataFrame:
    risk_component = builder.components.get_component(risk)
    return risk_component.get_data(
        builder, builder.configuration[risk_component.name]["data_sources"]["exposure"]
    )


###############################
# Relative risk data handlers #
###############################


def rebin_relative_risk_data(
    builder, risk: EntityString, relative_risk_data: pd.DataFrame
) -> pd.DataFrame:
    """Rebin relative risk data if necessary.

    When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
    After rebinning, rr for both exposed and unexposed categories should be the weighted sum of relative risk
    of the component categories where weights are relative proportions of exposure of those categories.
    For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
    for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
    (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
    """
    if not risk in builder.configuration.to_dict():
        return relative_risk_data

    rebin_exposed_categories = set(builder.configuration[risk]["rebinned_exposed"])

    if rebin_exposed_categories:
        # todo make sure this works
        exposure_data = load_exposure_data(builder, risk)
        relative_risk_data = _rebin_relative_risk_data(
            relative_risk_data, exposure_data, rebin_exposed_categories
        )

    return relative_risk_data


def _rebin_relative_risk_data(
    relative_risk_data: pd.DataFrame,
    exposure_data: pd.DataFrame,
    rebin_exposed_categories: set,
) -> pd.DataFrame:
    cols = list(exposure_data.columns.difference(["value"]))

    relative_risk_data = relative_risk_data.merge(exposure_data, on=cols)
    relative_risk_data["value_x"] = relative_risk_data.value_x.multiply(
        relative_risk_data.value_y
    )
    relative_risk_data.parameter = relative_risk_data["parameter"].map(
        lambda p: "cat1" if p in rebin_exposed_categories else "cat2"
    )
    relative_risk_data = relative_risk_data.groupby(cols).sum().reset_index()
    relative_risk_data["value"] = relative_risk_data.value_x.divide(
        relative_risk_data.value_y
    ).fillna(0)
    return relative_risk_data.drop(columns=["value_x", "value_y"])


##############
# Validators #
##############


def validate_distribution_data_source(builder: Builder, risk: EntityString) -> None:
    """Checks that the exposure distribution specification is valid."""
    exposure_type = builder.configuration[risk]["data_sources"]["exposure"]
    rebin = builder.configuration[risk]["rebinned_exposed"]
    category_thresholds = builder.configuration[risk]["category_thresholds"]

    if risk.type == "alternative_risk_factor":
        if exposure_type != "data" or rebin:
            raise ValueError(
                "Parameterized risk components are not available for alternative risks."
            )

        if not category_thresholds:
            raise ValueError("Must specify category thresholds to use alternative risks.")

    elif risk.type not in ["risk_factor", "coverage_gap"]:
        raise ValueError(f"Unknown risk type {risk.type} for risk {risk.name}")


def validate_relative_risk_data_source(builder, risk: EntityString, target: TargetString):
    from vivarium_public_health.risks import RiskEffect

    source_key = RiskEffect.get_name(risk, target)
    source_config = builder.configuration[source_key]

    provided_keys = set(
        k
        for k, v in source_config["distribution_args"].to_dict().items()
        if isinstance(v, (int, float))
    )

    source_map = {
        "data": set(),
        "relative risk value": {"relative_risk"},
        "normal distribution": {"mean", "se"},
        "log distribution": {"log_mean", "log_se", "tau_squared"},
    }

    if provided_keys not in source_map.values():
        raise ValueError(
            f"The acceptable parameter options for specifying relative risk are: "
            f"{source_map.values()}. You provided {provided_keys} for {source_key}."
        )

    source_type = [k for k, v in source_map.items() if provided_keys == v][0]

    if source_type == "relative risk value":
        if not 1 <= source_type <= 100:
            raise ValueError(
                "If specifying a single value for relative risk, it should be in the range [1, 100]. "
                f"You provided {source_type} for {source_key}."
            )
    elif source_type == "normal distribution":
        if source_config["mean"] <= 0 or source_config["se"] <= 0:
            raise ValueError(
                f"To specify parameters for a normal distribution for a risk effect, you must provide"
                f"both mean and se above 0. This is not the case for {source_key}."
            )
    elif source_type == "log distribution":
        if source_config["log_mean"] <= 0 or source_config["log_se"] <= 0:
            raise ValueError(
                f"To specify parameters for a log distribution for a risk effect, you must provide"
                f"both log_mean and log_se above 0. This is not the case for {source_key}."
            )
        if source_config["tau_squared"] < 0:
            raise ValueError(
                f"To specify parameters for a log distribution for a risk effect, you must provide"
                f"tau_squared >= 0. This is not the case for {source_key}."
            )
    else:
        pass

    return source_type


def validate_relative_risk_rebin_source(
    builder, risk: EntityString, target: TargetString, data: pd.DataFrame
):
    if data.index.size == 0:
        raise ValueError(
            f"Subsetting {risk} relative risk data to {target.name} {target.measure} "
            "returned an empty DataFrame. Check your artifact."
        )
    if risk in builder.configuration.to_dict():
        validate_rebin_source(builder, risk, data)


def validate_rebin_source(builder, risk: EntityString, data: pd.DataFrame) -> None:

    if not isinstance(data, pd.DataFrame):
        return

    rebin_exposed_categories = set(builder.configuration[risk]["rebinned_exposed"])

    if rebin_exposed_categories and builder.configuration[risk]["category_thresholds"]:
        raise ValueError(
            f"Rebinning and category thresholds are mutually exclusive. "
            f"You provided both for {risk.name}."
        )

    if rebin_exposed_categories and "polytomous" not in builder.data.load(
        f"{risk}.distribution"
    ):
        raise ValueError(
            f"Rebinning is only supported for polytomous risks. You provided "
            f"rebinning exposed categoriesfor {risk.name}, which is of "
            f"type {builder.data.load(f'{risk}.distribution')}."
        )

    invalid_cats = rebin_exposed_categories.difference(set(data.parameter))
    if invalid_cats:
        raise ValueError(
            f"The following provided categories for the rebinned exposed "
            f"category of {risk.name} are not found in the exposure data: "
            f"{invalid_cats}."
        )

    if rebin_exposed_categories == set(data.parameter):
        raise ValueError(
            f"The provided categories for the rebinned exposed category of "
            f"{risk.name} comprise all categories for the exposure data. "
            f"At least one category must be left out of the provided categories "
            f"to be rebinned into the unexposed category."
        )
