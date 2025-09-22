from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from vivarium import Component as _Component
from vivarium.framework.engine import SimulationContext

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData


INDEX = pd.Index([4, 8, 15, 16, 23, 42])


@pytest.mark.parametrize(
    "source, expected_return",
    [
        (lambda idx: pd.Series(1.0, index=INDEX), pd.Series(1.0, index=INDEX)),
        (["attr1", "attr2"], pd.DataFrame({"attr1": [10.0], "attr2": [20.0]}, index=INDEX)),
        (42, None),  # should raise
    ],
)
def test_source_callable(
    source: pd.Series[float] | list[str] | int,
    expected_return: pd.Series[float] | pd.DataFrame | None,
) -> None:
    """Test that the source is correctly converted to a callable if needed."""

    class Component(_Component):
        @property
        def columns_created(self) -> list[str]:
            return ["attr1", "attr2"]

        def setup(self, builder: Builder) -> None:
            self.attribute_pipeline = builder.value.register_attribute_producer(
                "some-attribute",
                source=source,  # type: ignore [arg-type] # we are testing invalid types too
                component=self,
            )

        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            update = pd.DataFrame({"attr1": [10.0], "attr2": [20.0]}, index=pop_data.index)
            self.population_view.update(update)

    sim = SimulationContext(components=[Component()])
    sim.setup()
    sim.initialize_simulants()
    pl = sim._values.get_attribute("some-attribute")
    if expected_return is not None:
        attribute = pl(INDEX)
        assert type(attribute) == type(expected_return)
        if isinstance(expected_return, pd.DataFrame) and isinstance(attribute, pd.DataFrame):
            pd.testing.assert_frame_equal(attribute, expected_return)
        elif isinstance(expected_return, pd.Series) and isinstance(attribute, pd.Series):
            assert attribute.equals(expected_return)
    else:
        with pytest.raises(
            TypeError,
            match=(
                "The source of an attribute pipeline must be a callable or a list "
                f"of column names, but got {type(source)}."
            ),
        ):
            pl(INDEX)
