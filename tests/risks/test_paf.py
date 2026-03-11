"""
==========================================
Calibration Constant Pipeline Infrastructure
==========================================

Tests for :mod:`vivarium_public_health.risks.paf`.

These tests validate the helper functions and internal machinery that register
pipelines whose values are reduced by a joint calibration constant.
The module under test provides two public entry-points:

* ``register_risk_affected_rate_producer``   (``is_rate=True``)
* ``register_risk_affected_attribute_producer`` (``is_rate=False``)

Both delegate to ``_RiskAffectedPipeline``, which:

1. Creates a calibration constant value-producer pipeline
   (``<name>.calibration_constant``) with a custom combiner (list-append)
   and post-processor (``raw_union_post_processor``).
2. Registers the target pipeline (rate or attribute) with a post-processor
   that applies ``value * (1 - calibration_constant)`` to non-zero rows.
3. In ``on_post_setup``, precomputes the calibration constant into a lookup
   table so that it is available without re-running the full pipeline each
   time-step.

Redundancy notes
----------------
* ``test_risk_deletion`` in ``tests/disease/test_disease.py`` exercises the
  rate-producer path with a single calibration constant modifier via a full
  ``DiseaseModel`` stack.  The tests below cover the *same* code path with
  minimal wrappers so that failures pinpoint ``paf.py`` rather than disease
  infrastructure.
* ``test_incidence`` in the same file implicitly tests the zero calibration
  constant baseline (no modifier registered).  We cover that explicitly here
  as well.
"""

import numpy as np
import pandas as pd
from vivarium import Component, InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.utilities import from_yearly

from tests.test_utilities import build_table_with_age
from vivarium_public_health.disease import DiseaseModel, DiseaseState, RateTransition
from vivarium_public_health.disease.state import SusceptibleState
from vivarium_public_health.population import BasePopulation
from vivarium_public_health.risks.paf import (
    get_calibration_constant_pipeline_name,
    register_risk_affected_attribute_producer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AttributeSource(Component):
    """Minimal component that exposes a named attribute pipeline via
    ``register_risk_affected_attribute_producer``.

    The pipeline simply returns a constant ``pd.Series`` for every simulant.
    """

    CONFIGURATION_DEFAULTS = {
        "_attribute_source": {
            "data_sources": {
                "base_value": "base_value_data_source",
            },
        },
    }

    def __init__(self, pipeline_name: str, base_value: float):
        super().__init__()
        self._pipeline_name = pipeline_name
        self._base_value = base_value

    @property
    def name(self) -> str:
        return f"attribute_source.{self._pipeline_name}"

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
        register_risk_affected_attribute_producer(
            builder=builder,
            name=self._pipeline_name,
            source=self.base_value_table,
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


def _make_disease_model_components(
    base_rate: float,
    key: str = "sequela.test_cause.incidence_rate",
):
    """Return ``(components_list, transition_rate_pipeline_name)`` for a
    bare-minimum disease model that uses ``RateTransition``."""
    healthy = SusceptibleState("healthy")
    sick = DiseaseState("sick")
    transition = RateTransition(
        input_state=healthy,
        output_state=sick,
        transition_rate=key,
        rate_type="incidence_rate",
    )
    healthy.transition_set.append(transition)
    model = DiseaseModel("test", residual_state=healthy, states=[healthy, sick])
    return [BasePopulation(), model], "sick.incidence_rate"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGetCalibrationConstantPipelineName:
    def test_basic(self):
        assert (
            get_calibration_constant_pipeline_name("foo.bar")
            == "foo.bar.calibration_constant"
        )

    def test_empty(self):
        assert get_calibration_constant_pipeline_name("") == ".calibration_constant"


# ---------------------------------------------------------------------------
# Integration tests — rate producer
# ---------------------------------------------------------------------------


class TestRateProducer:
    """Exercise ``register_risk_affected_rate_producer`` through the
    ``RateTransition`` component that delegates to it."""

    def test_no_calibration_constant_modifier(self, base_config, base_plugins):
        """When no calibration constant modifier is registered the pipeline
        should return the base rate unmodified (calibration constant = 0)."""
        base_rate = 0.7
        time_step = pd.Timedelta(days=base_config.time.step_size)
        key = "sequela.test_cause.incidence_rate"
        components, pipeline_name = _make_disease_model_components(base_rate, key)

        sim = InteractiveContext(
            components=components,
            configuration=base_config,
            plugin_configuration=base_plugins,
            setup=False,
        )
        sim._data.write(key, base_rate)
        sim.setup()
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        expected = from_yearly(base_rate, time_step)
        assert np.allclose(actual, expected, atol=1e-5)

    def test_single_calibration_constant_modifier(self, base_config, base_plugins):
        """A single modifier with calibration constant = c should yield
        rate * (1 - c)."""
        base_rate = 0.7
        calibration_value = 0.1
        time_step = pd.Timedelta(days=base_config.time.step_size)
        key = "sequela.test_cause.incidence_rate"
        components, pipeline_name = _make_disease_model_components(base_rate, key)
        components.append(_CalibrationConstantModifier(pipeline_name, calibration_value))

        sim = InteractiveContext(
            components=components,
            configuration=base_config,
            plugin_configuration=base_plugins,
            setup=False,
        )
        sim._data.write(key, base_rate)
        sim.setup()
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        expected = from_yearly(base_rate * (1 - calibration_value), time_step)
        assert np.allclose(actual, expected, atol=1e-5)

    def test_multiple_calibration_constant_modifiers(self, base_config, base_plugins):
        """Two independent calibration constant modifiers (c1, c2) should
        combine as joint = 1 - (1 - c1) * (1 - c2)."""
        base_rate = 0.7
        c1 = 0.1
        c2 = 0.2
        time_step = pd.Timedelta(days=base_config.time.step_size)
        key = "sequela.test_cause.incidence_rate"
        components, pipeline_name = _make_disease_model_components(base_rate, key)
        components.append(_CalibrationConstantModifier(pipeline_name, c1))
        components.append(_CalibrationConstantModifier(pipeline_name, c2))

        sim = InteractiveContext(
            components=components,
            configuration=base_config,
            plugin_configuration=base_plugins,
            setup=False,
        )
        sim._data.write(key, base_rate)
        sim.setup()
        sim.step()

        joint_calibration_constant = 1 - (1 - c1) * (1 - c2)
        actual = sim.get_population(pipeline_name).squeeze()
        expected = from_yearly(base_rate * (1 - joint_calibration_constant), time_step)
        assert np.allclose(actual, expected, atol=1e-5)

    def test_calibration_constant_of_one_zeroes_rate(self, base_config, base_plugins):
        """Calibration constant = 1 should zero out the rate completely."""
        base_rate = 0.7
        calibration_value = 1.0
        key = "sequela.test_cause.incidence_rate"
        components, pipeline_name = _make_disease_model_components(base_rate, key)
        components.append(_CalibrationConstantModifier(pipeline_name, calibration_value))

        sim = InteractiveContext(
            components=components,
            configuration=base_config,
            plugin_configuration=base_plugins,
            setup=False,
        )
        sim._data.write(key, base_rate)
        sim.setup()
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        assert np.allclose(actual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Integration tests — attribute producer
# ---------------------------------------------------------------------------


class TestAttributeProducer:
    """Exercise ``register_risk_affected_attribute_producer`` through the
    minimal ``_AttributeSource`` wrapper."""

    def test_no_calibration_constant_modifier(self, base_config, base_plugins):
        """No modifier → attribute returned verbatim
        (calibration constant = 0)."""
        base_value = 0.5
        pipeline_name = "test_attribute"
        source = _AttributeSource(pipeline_name, base_value)

        sim = InteractiveContext(
            components=[BasePopulation(), source],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        assert np.allclose(actual, base_value, atol=1e-5)

    def test_single_calibration_constant_modifier(self, base_config, base_plugins):
        """Attribute producer with calibration constant = c → value * (1 - c)."""
        base_value = 0.5
        calibration_value = 0.25
        pipeline_name = "test_attribute"
        source = _AttributeSource(pipeline_name, base_value)
        modifier = _CalibrationConstantModifier(pipeline_name, calibration_value)

        sim = InteractiveContext(
            components=[BasePopulation(), source, modifier],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        expected = base_value * (1 - calibration_value)
        assert np.allclose(actual, expected, atol=1e-5)

    def test_multiple_calibration_constant_modifiers(self, base_config, base_plugins):
        """Two calibration constants on an attribute producer combine via
        union formula."""
        base_value = 100.0
        c1 = 0.1
        c2 = 0.3
        pipeline_name = "test_attribute"
        source = _AttributeSource(pipeline_name, base_value)

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

        joint_calibration_constant = 1 - (1 - c1) * (1 - c2)
        actual = sim.get_population(pipeline_name).squeeze()
        expected = base_value * (1 - joint_calibration_constant)
        assert np.allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# _apply_calibration_constant edge cases
# ---------------------------------------------------------------------------


class TestApplyCalibrationConstant:
    """Test the ``_apply_calibration_constant`` post-processor's behaviour
    with mixed zero / non-zero base values."""

    def test_all_zero_base_values(self, base_config, base_plugins):
        """When every simulant has a zero base value, calibration constant
        should have no effect and the result should remain all zeros (the
        short-circuit branch in ``_apply_calibration_constant``)."""
        base_value = 0.0
        calibration_value = 0.5
        pipeline_name = "test_attribute"
        source = _AttributeSource(pipeline_name, base_value)
        modifier = _CalibrationConstantModifier(pipeline_name, calibration_value)

        sim = InteractiveContext(
            components=[BasePopulation(), source, modifier],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name).squeeze()
        assert np.allclose(actual, 0.0, atol=1e-10)

    def test_mixed_zero_and_nonzero(self, base_config, base_plugins):
        """Calibration constant should only be applied to non-zero rows;
        zero rows stay zero."""
        calibration_value = 0.5
        pipeline_name = "test_mixed"
        nonzero_value = 10.0

        class _MixedSource(Component):
            """Returns ``nonzero_value`` for the first half of simulants and
            0 for the rest."""

            @property
            def name(self) -> str:
                return "mixed_source"

            def setup(self, builder: Builder) -> None:
                register_risk_affected_attribute_producer(
                    builder=builder,
                    name=pipeline_name,
                    source=self._source,
                )

            def _source(self, index: pd.Index) -> pd.Series:
                values = pd.Series(0.0, index=index)
                half = len(index) // 2
                values.iloc[:half] = nonzero_value
                return values

        sim = InteractiveContext(
            components=[
                BasePopulation(),
                _MixedSource(),
                _CalibrationConstantModifier(pipeline_name, calibration_value),
            ],
            configuration=base_config,
            plugin_configuration=base_plugins,
        )
        sim.step()

        actual = sim.get_population(pipeline_name)
        pop_size = len(actual)
        half = pop_size // 2

        # Non-zero rows should be reduced by calibration constant
        assert np.allclose(
            actual.iloc[:half].squeeze(),
            nonzero_value * (1 - calibration_value),
            atol=1e-5,
        )
        # Zero rows should remain zero
        assert np.allclose(actual.iloc[half:].squeeze(), 0.0, atol=1e-10)
