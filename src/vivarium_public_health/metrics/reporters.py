from pathlib import Path
from typing import NamedTuple, Optional, Union

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


def write_dataframe(
    results: pd.DataFrame,
    measure: str,
    results_dir: Optional[Union[str, Path]],
) -> None:
    """Utility function for observation 'report' methods to write pd.DataFrames to parquet"""
    if results_dir is None:
        raise ValueError("A results_dir must be specified to write out results.")
    results.to_parquet(Path(results_dir) / f"{measure}.parquet", index=False)
