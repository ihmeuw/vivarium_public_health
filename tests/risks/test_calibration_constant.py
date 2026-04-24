"""
Tests for :mod:`vivarium_public_health.risks.calibration_constant`.

Redundancy notes
----------------
* ``test_risk_deletion`` in ``tests/disease/test_disease.py`` exercises the
  rate-producer path with a single calibration constant modifier via a full
  ``DiseaseModel`` stack.  The tests below cover the *same* code path with
  minimal wrappers so that failures pinpoint ``calibration_constant.py``
  rather than disease infrastructure.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from vivarium import Component, InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.utilities import from_yearly
from vivarium.framework.values import AttributePostProcessor, ValuesManager

from tests.test_utilities import build_table_with_age
from vivarium_public_health.causal_factor.calibration_constant import (
    get_calibration_constant_pipeline_name,
    register_risk_affected_attribute_producer,
    register_risk_affected_rate_producer,
)
from vivarium_public_health.population import BasePopulation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AttributeSource(Component):
    """Minimal component that exposes a named pipeline via
    ``register_risk_affected_attribute_producer`` or
    ``register_risk_affected_rate_producer``.

    The pipeline simply returns a constant ``pd.Series`` for every simulant.
    """

    CONFIGURATION_DEFAULTS = {
        "_attribute_source": {
            "data_sources": {
                "base_value": "base_value_data_source",
            },
        },
    }

    def __init__(
        self,
        pipeline_name: str,
        base_value: float,
        is_rate: bool = False,
        additional_post_processors: (
            AttributePostProcessor | Sequence[AttributePostProcessor]
        ) = (),
    ):
        super().__init__()
        self._pipeline_name = pipeline_name
        self._base_value = base_value
        self._is_rate = is_rate
        self._additional_post_processors = additional_post_processors

    @property
    def name(self) -> str:
        kind = "rate" if self._is_rate else "attribute"
        return f"attribute_source.{kind}.{self._pipeline_name}"

    @property
    def configuration_defaults(self) -> dict:
        return {
            self.name: {
                "data_sources": {
                    "base_value": self._base_value,
                },
            },
        }

    def setup(self, builder: Builder) -> None:
        self.base_value_table = self.build_lookup_table(builder, "base_value")
        register_fn = (
            register_risk_affected_rate_producer
            if self._is_rate
            else register_risk_affected_attribute_producer
        )
        register_fn(
            builder=builder,
            name=self._pipeline_name,
            source=self.base_value_table,
            additional_post_processors=self._additional_post_processors,
        )


class _CalibrationConstantModifier(Component):
    """Registers a value modifier on ``<target>.calibration_constant``
    with a constant calibration constant."""

    def __init__(self, target_pipeline: str, calibration_value: float):
        super().__init__()
        self._target_pipeline = target_pipeline
        self._calibration_value = calibration_value

    @property
    def name(self) -> str:
        return (
            f"calibration_constant_modifier"
            f"({self._target_pipeline}, {self._calibration_value})"
        )

    def setup(self, builder: Builder) -> None:
        year_start = builder.configuration.time.start.year
        year_end = builder.configuration.time.end.year
        data = build_table_with_age(
            self._calibration_value, parameter_columns={"year": (year_start, year_end)}
        )
        builder.value.register_value_modifier(
            get_calibration_constant_pipeline_name(self._target_pipeline),
            modifier=lambda: data,
        )


# ---------------------------------------------------------------------------
# Helpers — post-processor functions
# ---------------------------------------------------------------------------


def _double_post_processor(
    index: pd.Index, value: pd.Series, manager: ValuesManager
) -> pd.Series:
    """Multiply values by 2."""
    return value * 2


def _add_one_post_processor(
    index: pd.Index, value: pd.Series, manager: ValuesManager
) -> pd.Series:
    """Add 1 to all values."""
    return value + 1


# ---------------------------------------------------------------------------
# Integration tests — attribute / rate producer via _AttributeSource
# ---------------------------------------------------------------------------


class TestProducer:
    """Exercise ``register_risk_affected_attribute_producer`` and
    ``register_risk_affected_rate_producer`` through the minimal
    ``_AttributeSource`` wrapper, parameterized over producer type and
    post-processors."""

    _POST_PROCESSOR_PARAMS = [
        pytest.param(((), lambda v: v), id="no_pp"),
        pytest.param((_double_post_processor, lambda v: v * 2), id="double"),
        pytest.param(
            ([_double_post_processor, _add_one_post_processor], lambda v: v * 2 + 1),
            id="double_then_add_one",
        ),
    ]

    @pytest.fixture(params=[False, True], ids=["attribute", "rate"])
    def is_rate(self, request):
        return request.param

    @pytest.fixture(params=_POST_PROCESSOR_PARAMS)
    def post_processor(self, request):
        """Return ``(post_processors, expected_transform)``."""
        return request.param

    def _expected(self, base, calibration, is_rate, time_step, transform):
        """Compute expected pipeline output."""
        value = base * (1 - calibration)
        if is_rate:
            value = from_yearly(value, time_step)
        return transform(value)

    def test_no_calibration_constant_modifier(
        self, base_config, base_plugins, is_rate, post_processor
    ):
        """No modifier → base value returned (calibration constant = 0),
        then post-processors applied."""
        post_processors, transform = post_processor
        base_value = 0.7 if is_rate else 10.0
        time_step = pd.Timedelta(days=base_config.time.step_size)
        pipeline_name = "test_pipeline"
        source = _AttributeSource(
            pipeline_name,
            base_value,
            is_rate=is_rate,
            additional_post_processors=post_processors,
        )

        sim = InteractiveContext(
            components=[BasePopulation(), source],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        expected = self._expected(base_value, 0, is_rate, time_step, transform)
        assert np.allclose(actual, expected, atol=1e-5)

    def test_single_calibration_constant_modifier(
        self, base_config, base_plugins, is_rate, post_processor
    ):
        """Single calibration constant = c → value * (1 - c),
        then post-processors applied."""
        post_processors, transform = post_processor
        base_value = 0.7 if is_rate else 10.0
        calibration_value = 0.25
        time_step = pd.Timedelta(days=base_config.time.step_size)
        pipeline_name = "test_pipeline"
        source = _AttributeSource(
            pipeline_name,
            base_value,
            is_rate=is_rate,
            additional_post_processors=post_processors,
        )
        modifier = _CalibrationConstantModifier(pipeline_name, calibration_value)

        sim = InteractiveContext(
            components=[BasePopulation(), source, modifier],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        expected = self._expected(
            base_value, calibration_value, is_rate, time_step, transform
        )
        assert np.allclose(actual, expected, atol=1e-5)

    def test_multiple_calibration_constant_modifiers(
        self, base_config, base_plugins, is_rate, post_processor
    ):
        """Two calibration constants combine via union formula,
        then post-processors applied."""
        post_processors, transform = post_processor
        base_value = 0.7 if is_rate else 10.0
        c1, c2 = 0.1, 0.3
        time_step = pd.Timedelta(days=base_config.time.step_size)
        pipeline_name = "test_pipeline"
        source = _AttributeSource(
            pipeline_name,
            base_value,
            is_rate=is_rate,
            additional_post_processors=post_processors,
        )

        sim = InteractiveContext(
            components=[
                BasePopulation(),
                source,
                _CalibrationConstantModifier(pipeline_name, c1),
                _CalibrationConstantModifier(pipeline_name, c2),
            ],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        joint_calibration = 1 - (1 - c1) * (1 - c2)
        actual = sim.get_population(pipeline_name).squeeze()
        expected = self._expected(
            base_value, joint_calibration, is_rate, time_step, transform
        )
        assert np.allclose(actual, expected, atol=1e-5)

    def test_calibration_constant_of_one_zeroes(self, base_config, base_plugins, is_rate):
        """Calibration constant = 1 should zero out the value completely."""
        base_value = 0.7 if is_rate else 10.0
        pipeline_name = "test_pipeline"
        source = _AttributeSource(pipeline_name, base_value, is_rate=is_rate)
        modifier = _CalibrationConstantModifier(pipeline_name, 1.0)

        sim = InteractiveContext(
            components=[BasePopulation(), source, modifier],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        assert np.allclose(actual, 0.0, atol=1e-10)
