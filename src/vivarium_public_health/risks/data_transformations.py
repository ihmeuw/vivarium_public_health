"""
=========================
Risk Data Transformations
=========================

This module contains tools for handling raw risk exposure and relative
risk data and performing any necessary data transformations.

"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from vivarium.framework.engine import Builder

from vivarium_public_health.utilities import EntityString, TargetString

#############
# Utilities #
#############


def is_data_from_artifact(data_source: str) -> bool:
    return isinstance(data_source, str) and "::" not in data_source


def pivot_categorical(
    builder: Builder, risk: Optional[EntityString], data: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Pivots data that is long on categories to be wide."""
    # todo remove if statement when relative risk has been updated
    if risk:
        # todo remove dependency on artifact manager having exactly one value column
        value_column = builder.data.value_columns()(f"{risk}.exposure")[0]
        pivot_column = "parameter"
        index_cols = [
            column for column in data.columns if column not in [value_column, pivot_column]
        ]
        data = data.pivot_table(
            index=index_cols, columns=pivot_column, values=value_column
        ).reset_index()
        data.columns.name = None
    else:
        index_cols = ["sex", "age_start", "age_end", "year_start", "year_end"]
        index_cols = [k for k in index_cols if k in data.columns]
        data = data.pivot_table(
            index=index_cols, columns="parameter", values="value"
        ).reset_index()
        data.columns.name = None

    value_columns = [column for column in data.columns if column not in index_cols]
    return data, value_columns


##########################
# Exposure data handlers #
##########################


def get_distribution_data(builder, risk: EntityString) -> Dict[str, Any]:
    validate_distribution_data_source(builder, risk)
    data = load_distribution_data(builder, risk)
    validate_distribution_data(data)
    return data


def get_exposure_post_processor(builder, risk: EntityString):
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


def load_distribution_data(builder: Builder, risk: EntityString) -> Dict[str, Any]:
    distribution_type = get_distribution_type(builder, risk)
    exposure_data, value_columns = get_exposure_data(builder, risk, distribution_type)

    data = {
        "distribution_type": distribution_type,
        "exposure": exposure_data,
        "exposure_value_columns": value_columns,
        "exposure_standard_deviation": get_exposure_standard_deviation_data(
            builder, risk, distribution_type
        ),
        "weights": get_exposure_distribution_weights(builder, risk, distribution_type),
    }
    return data


def get_distribution_type(builder, risk: EntityString):
    risk_config = builder.configuration[risk]
    exposure = risk_config["data_sources"]["exposure"]

    if is_data_from_artifact(exposure) and not risk_config["rebinned_exposed"]:
        distribution_type = builder.data.load(f"{risk}.distribution")
    else:
        distribution_type = "dichotomous"

    return distribution_type


def get_exposure_data(
    builder, risk: EntityString, distribution_type: str
) -> Tuple[pd.DataFrame, List[str]]:
    exposure_data = load_exposure_data(builder, risk)
    exposure_data = rebin_exposure_data(builder, risk, exposure_data)

    if distribution_type in [
        "dichotomous",
        "ordered_polytomous",
        "unordered_polytomous",
        "lbwsg",
    ]:
        exposure_data, value_columns = pivot_categorical(builder, risk, exposure_data)
    else:
        value_columns = builder.data.value_columns()(f"{risk}.exposure")

    return exposure_data, value_columns


def load_exposure_data(builder: Builder, risk: EntityString) -> pd.DataFrame:
    risk_component = builder.components.get_component(risk)
    exposure_data = risk_component.get_data(
        builder, builder.configuration[risk_component.name]["data_sources"]["exposure"]
    )

    if isinstance(exposure_data, (int, float)):
        value_columns = builder.data.value_columns()(f"{risk}.exposure")
        cat1 = builder.data.load("population.demographic_dimensions")
        cat1["parameter"] = "cat1"
        cat1[value_columns] = float(exposure_data)

        cat2 = cat1.copy()
        cat2["parameter"] = "cat2"
        cat2[value_columns] = 1 - cat2["value"]

        exposure_data = pd.concat([cat1, cat2], ignore_index=True)

    return exposure_data


def get_exposure_standard_deviation_data(
    builder: Builder, risk: EntityString, distribution_type: str
) -> Union[pd.DataFrame, None]:
    if distribution_type not in ["normal", "lognormal", "ensemble"]:
        return None
    data_source = builder.configuration[risk]["data_sources"]["exposure_standard_deviation"]
    return builder.data.load(data_source)


def get_exposure_distribution_weights(
    builder: Builder, risk: EntityString, distribution_type: str
) -> Union[Tuple[pd.DataFrame, List[str]], None]:
    if distribution_type != "ensemble":
        return None

    data_source = builder.configuration[risk]["data_source"]["ensemble_distribution_weights"]
    weights = builder.data.load(data_source)
    weights, distributions = pivot_categorical(builder, risk, weights)
    if "glnorm" in weights.columns:
        if np.any(weights["glnorm"]):
            raise NotImplementedError("glnorm distribution is not supported")
        weights = weights.drop(columns=["glnorm"])
    return weights, distributions


def rebin_exposure_data(builder, risk: EntityString, exposure_data: pd.DataFrame):
    validate_rebin_source(builder, risk, exposure_data)
    rebin_exposed_categories = set(builder.configuration[risk]["rebinned_exposed"])

    if rebin_exposed_categories:
        exposure_data = _rebin_exposure_data(exposure_data, rebin_exposed_categories)

    return exposure_data


def _rebin_exposure_data(
    exposure_data: pd.DataFrame, rebin_exposed_categories: set
) -> pd.DataFrame:
    exposure_data["parameter"] = exposure_data["parameter"].map(
        lambda p: "cat1" if p in rebin_exposed_categories else "cat2"
    )
    return (
        exposure_data.groupby(list(exposure_data.columns.difference(["value"])))
        .sum()
        .reset_index()
    )


###############################
# Relative risk data handlers #
###############################


def get_relative_risk_data(builder, risk: EntityString, target: TargetString):
    source_type = validate_relative_risk_data_source(builder, risk, target)
    relative_risk_data = load_relative_risk_data(builder, risk, target, source_type)
    validate_relative_risk_rebin_source(builder, risk, target, relative_risk_data)
    relative_risk_data = rebin_relative_risk_data(builder, risk, relative_risk_data)

    if get_distribution_type(builder, risk) in [
        "dichotomous",
        "ordered_polytomous",
        "unordered_polytomous",
    ]:
        relative_risk_data, _ = pivot_categorical(builder, None, relative_risk_data)
        # Check if any values for relative risk are below expected boundary of 1.0
        category_columns = [c for c in relative_risk_data.columns if "cat" in c]
        if not relative_risk_data[
            (relative_risk_data[category_columns] < 1.0).any(axis=1)
        ].empty:
            logger.warning(
                f"WARNING: Some data values are below the expected boundary of 1.0 for {risk}.relative_risk"
            )

    else:
        relative_risk_data = relative_risk_data.drop(columns=["parameter"])

    return relative_risk_data


def load_relative_risk_data(
    builder: Builder, risk: EntityString, target: TargetString, source_type: str
):
    from vivarium_public_health.risks import RiskEffect

    source_key = RiskEffect.get_name(risk, target)
    relative_risk_source = builder.configuration[source_key]["data_sources"]["relative_risk"]

    if source_type == "data":
        relative_risk_data = builder.data.load(f"{risk}.relative_risk")
        correct_target = (relative_risk_data["affected_entity"] == target.name) & (
            relative_risk_data["affected_measure"] == target.measure
        )
        relative_risk_data = relative_risk_data[correct_target].drop(
            columns=["affected_entity", "affected_measure"]
        )

    elif source_type == "relative risk value":
        relative_risk_data = _make_relative_risk_data(
            builder, float(relative_risk_source["relative_risk"])
        )

    else:  # distribution
        parameters = {
            k: v for k, v in relative_risk_source.to_dict().items() if v is not None
        }
        random_state = np.random.RandomState(
            builder.randomness.get_seed(
                f"effect_of_{risk.name}_on_{target.name}.{target.measure}"
            )
        )
        cat1_value = generate_relative_risk_from_distribution(random_state, parameters)
        relative_risk_data = _make_relative_risk_data(builder, cat1_value)

    return relative_risk_data


def generate_relative_risk_from_distribution(
    random_state: np.random.RandomState, parameters: dict
) -> Union[float, pd.Series, np.ndarray]:
    first = pd.Series(list(parameters.values())[0])
    length = len(first)
    index = first.index

    for v in parameters.values():
        if length != len(pd.Series(v)) or not index.equals(pd.Series(v).index):
            raise ValueError(
                "If specifying vectorized parameters, all parameters "
                "must be the same length and have the same index."
            )

    if "mean" in parameters:  # normal distribution
        rr_value = random_state.normal(parameters["mean"], parameters["se"])
    elif "log_mean" in parameters:  # log distribution
        log_value = parameters["log_mean"] + parameters["log_se"] * random_state.randn()
        if parameters["tau_squared"]:
            log_value += random_state.normal(0, parameters["tau_squared"])
        rr_value = np.exp(log_value)
    else:
        raise NotImplementedError(
            f"Only normal distributions (supplying mean and se) and log distributions "
            f"(supplying log_mean, log_se, and tau_squared) are currently supported."
        )

    rr_value = np.maximum(1, rr_value)

    return rr_value


def _make_relative_risk_data(builder, cat1_value: float) -> pd.DataFrame:
    cat1 = builder.data.load("population.demographic_dimensions")
    cat1["parameter"] = "cat1"
    cat1["value"] = cat1_value
    cat2 = cat1.copy()
    cat2["parameter"] = "cat2"
    cat2["value"] = 1
    return pd.concat([cat1, cat2], ignore_index=True)


def rebin_relative_risk_data(
    builder, risk: EntityString, relative_risk_data: pd.DataFrame
) -> pd.DataFrame:
    """When the polytomous risk is rebinned, matching relative risk needs to be rebinned.
    After rebinning, rr for both exposed and unexposed categories should be the weighted sum of relative risk
    of the component categories where weights are relative proportions of exposure of those categories.
    For example, if cat1, cat2, cat3 are exposed categories and cat4 is unexposed with exposure [0.1,0.2,0.3,0.4],
    for the matching rr = [rr1, rr2, rr3, 1], rebinned rr for the rebinned cat1 should be:
    (0.1 *rr1 + 0.2 * rr2 + 0.3* rr3) / (0.1+0.2+0.3)
    """
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


def get_exposure_effect(builder, risk: EntityString):
    distribution_type = get_distribution_type(builder, risk)
    risk_exposure = builder.value.get_value(f"{risk.name}.exposure")

    if distribution_type in ["normal", "lognormal", "ensemble"]:
        tmred = builder.data.load(f"{risk}.tmred")
        tmrel = 0.5 * (tmred["min"] + tmred["max"])
        scale = builder.data.load(f"{risk}.relative_risk_scalar")

        def exposure_effect(rates, rr):
            exposure = risk_exposure(rr.index)
            relative_risk = np.maximum(rr.values ** ((exposure - tmrel) / scale), 1)
            return rates * relative_risk

    else:

        def exposure_effect(rates, rr: pd.DataFrame) -> pd.Series:
            index_columns = ["index", risk.name]

            exposure = risk_exposure(rr.index).reset_index()
            exposure.columns = index_columns
            exposure = exposure.set_index(index_columns)

            relative_risk = rr.stack().reset_index()
            relative_risk.columns = index_columns + ["value"]
            relative_risk = relative_risk.set_index(index_columns)

            effect = relative_risk.loc[exposure.index, "value"].droplevel(risk.name)
            affected_rates = rates * effect
            return affected_rates

    return exposure_effect


##################################################
# Population attributable fraction data handlers #
##################################################


def get_population_attributable_fraction_data(
    builder: Builder, risk: EntityString, target: TargetString
):
    paf_data = builder.data.load(f"{risk}.population_attributable_fraction")
    if isinstance(paf_data, pd.DataFrame):
        correct_target = (paf_data["affected_entity"] == target.name) & (
            paf_data["affected_measure"] == target.measure
        )
        paf_data = paf_data[correct_target].drop(
            columns=["affected_entity", "affected_measure"]
        )
    else:
        exposure_data, exposure_value_cols = get_exposure_data(builder, risk)
        relative_risk_data, rr_value_cols = get_relative_risk_data(builder, risk, target)
        if set(exposure_value_cols) != set(rr_value_cols):
            error_msg = "Exposure and relative risk value columns must match. "
            missing_rr_cols = set(exposure_value_cols).difference(set(rr_value_cols))
            if missing_rr_cols:
                error_msg = error_msg + f"Missing relative risk columns: {missing_rr_cols}. "
            missing_exposure_cols = set(rr_value_cols).difference(set(exposure_value_cols))
            if missing_exposure_cols:
                error_msg = error_msg + f"Missing exposure columns: {missing_exposure_cols}. "
            raise ValueError(error_msg)
        # Build up dataframe for paf
        index_cols = [col for col in paf_data.columns if col not in exposure_value_cols]
        exposure_data = exposure_data.set_index(index_cols)
        relative_risk_data = relative_risk_data.set_index(index_cols)
        mean_rr = (exposure_data * relative_risk_data).sum(axis=1)
        paf_data = ((mean_rr - 1) / mean_rr).reset_index().rename(columns={0: "value"})
    return paf_data


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


def validate_distribution_data(distribution_data: Dict[str, Any]) -> None:
    exposure_data = distribution_data["exposure"]
    val_cols = distribution_data["exposure_value_columns"]
    if distribution_data["distribution_type"] == "dichotomous":
        if ((exposure_data[val_cols] < 0) | exposure_data[val_cols] > 1).any().any():
            raise ValueError(f"Exposure should be in the range [0, 1]")
    # TODO: validate that weights, standard deviation, and exposure have the
    #  same index cols for ensemble
    # TODO: validate that standard deviation and exposure have the same index
    #  cols for normal and lognormal
    # TODO: add more data validations


def validate_relative_risk_data_source(builder, risk: EntityString, target: TargetString):
    from vivarium_public_health.risks import RiskEffect

    source_key = RiskEffect.get_name(risk, target)
    source_config = builder.configuration[source_key]

    provided_keys = set(
        k
        for k, v in source_config["distribution_args"].to_dict().items()
        if isinstance(v, (int, float))
    )
    if isinstance(source_config["data_sources"]["relative_risk"], (float, int)):
        provided_keys.add("relative_risk")

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
        relative_risk_value = source_config["data_sources"]["relative_risk"]
        if not 1 <= relative_risk_value <= 100:
            raise ValueError(
                "If specifying a single value for relative risk, it should be in the range [1, 100]. "
                f"You provided {relative_risk_value} for {source_key}."
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
    validate_rebin_source(builder, risk, data)


def validate_rebin_source(builder, risk: EntityString, data: pd.DataFrame):
    rebin_exposed_categories = set(builder.configuration[risk.name]["rebinned_exposed"])

    if rebin_exposed_categories and builder.configuration[risk.name]["category_thresholds"]:
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
