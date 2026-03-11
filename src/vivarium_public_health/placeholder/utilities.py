"""
=========
Utilities
=========

This module contains utility functions for the placeholder components.

"""

import pandas as pd
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN

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
