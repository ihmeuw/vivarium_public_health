from typing import NamedTuple

from vivarium.framework.results import VALUE_COLUMN


class __Columns(NamedTuple):
    """column names"""

    VALUE: str = VALUE_COLUMN
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
