from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Union

import pandas as pd
from vivarium.framework.results import METRICS_COLUMN


class __Columns(NamedTuple):
    """column names"""

    VALUE: str = METRICS_COLUMN
    MEASURE: str = "measure"
    SEED: str = "random_seed"
    DRAW: str = "input_draw"
    TRANSITION: str = "transition"
    STATE: str = "state"
    ENTITY_TYPE: str = "entity_type"
    SUB_ENTITY: str = "sub_entity"
    ENTITY: str = "entity"

    @property
    def name(self) -> str:
        return "columns"


COLUMNS = __Columns()


def write_dataframe_to_parquet(
    results: pd.DataFrame,
    measure: str,
    entity_type: str,
    entity: str,
    sub_entity: Optional[str],
    results_dir: Optional[Union[str, Path]],
    random_seed: Optional[int],
    input_draw: Optional[int],
    output_filename: Optional[str] = None,
) -> None:
    """Utility function for observation 'report' methods to write pd.DataFrames to parquet"""
    if results_dir is None:
        raise ValueError("A results_dir must be specified to write out results.")
    results_dir = Path(results_dir)
    # Add extra cols
    col_mapper = {
        COLUMNS.MEASURE: measure,
        COLUMNS.ENTITY_TYPE: entity_type,
        COLUMNS.ENTITY: entity,
        COLUMNS.SUB_ENTITY: sub_entity,
        COLUMNS.SEED: random_seed,
        COLUMNS.DRAW: input_draw,
    }
    for col, val in col_mapper.items():
        results[col] = val

    # Sort the columns such that the stratifications (index) are first,
    # the value column is last, and sort the rows by the stratifications.
    other_cols = [c for c in results.columns if c != COLUMNS.VALUE]
    results = results[other_cols + [COLUMNS.VALUE]].sort_index().reset_index()

    # Concat and save
    output_filename = measure if not output_filename else output_filename
    results_file = results_dir / f"{output_filename}.parquet"
    if results_file.exists():
        # pd.to_parquet does not support an append mode
        original_results = pd.read_parquet(results_file)
        results = pd.concat([original_results, results], ignore_index=True)

    results.to_parquet(results_file, index=False)
