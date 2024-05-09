from pathlib import Path
from typing import Any, Dict, Optional, Union, NamedTuple

import pandas as pd


from vivarium.framework.results import METRICS_COLUMN


class __Columns(NamedTuple):
    """column names"""

    VALUE: str = METRICS_COLUMN
    MEASURE: str = "measure"
    SEED: str = "random_seed"
    DRAW: str = "input_draw"
    CAUSE: str = "cause"
    TRANSITION: str = "transition"
    STATE: str = "state"

    @property
    def name(self):
        return "columns"


COLUMNS = __Columns()


def write_dataframe_to_parquet(
    measure: str,
    results: pd.DataFrame,
    results_dir: Optional[Union[str, Path]],
    random_seed: Optional[int],
    input_draw: Optional[int],
    extra_cols: Dict[str, Any] = {},
) -> None:
    """Utility function for observation 'report' methods to write pd.DataFrames to parquet"""
    if results_dir is None:
        raise ValueError("A results_dir must be specified to write out results.")
    results_dir = Path(results_dir)
    # Add extra cols
    col_mapper = {
        **{COLUMNS.MEASURE: measure},
        **extra_cols,
        **{COLUMNS.SEED: random_seed, COLUMNS.DRAW: input_draw},
    }
    for col, val in col_mapper.items():
        if val is not None:
            results[col] = val
    # Sort the columns such that the stratifications (index) are first,
    # the value column is last, and sort the rows by the stratifications.
    other_cols = [c for c in results.columns if c != COLUMNS.VALUE]
    results = results[other_cols + [COLUMNS.VALUE]].sort_index().reset_index()

    # Concat and save
    results_file = results_dir / f"{measure}.parquet"
    if results_file.exists():
        # pd.to_parquet does not support an append mode
        original_results = pd.read_parquet(results_file)
        results = pd.concat([original_results, results], ignore_index=True)

    results.to_parquet(results_file, index=False)
