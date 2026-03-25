"""
=========
Utilities
=========

This module contains utility functions for the placeholder components.

"""

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN

from vivarium_public_health.utilities import EntityString

#############
# Utilities #
#############


def pivot_categorical(
    data: pd.DataFrame, pivot_column: str = "parameter", reset_index: bool = True
) -> pd.DataFrame:
    """Pivots data that is long on categories to be wide."""
    index_cols = [
        column
        for column in data.columns
        if column not in [DEFAULT_VALUE_COLUMN, pivot_column]
    ]
    data = data.pivot_table(
        index=index_cols, columns=pivot_column, values=DEFAULT_VALUE_COLUMN
    )
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
        post_processor = []

    return post_processor


def load_exposure_data(builder: Builder, risk: EntityString) -> pd.DataFrame:
    risk_component = builder.components.get_component(risk)
    return risk_component.get_data(
        builder, builder.configuration[risk_component.name]["data_sources"]["exposure"]
    )
