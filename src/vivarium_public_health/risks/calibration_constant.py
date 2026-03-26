"""
============================================
Calibration Constant Pipeline Infrastructure
============================================

This module provides helper functions and internal machinery that register
pipelines whose values are reduced by a joint calibration constant.
Population attributable fractions (PAFs) can often be used interchangeably
with calibration constants.

The module provides two public entry-points:

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

"""
from collections.abc import Callable, Sequence
from typing import Any
from typing import SupportsFloat as Numeric

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN
from vivarium.framework.resource import Resource
from vivarium.framework.values import (
    ValuesManager,
    multiplication_combiner,
    raw_union_post_processor,
)
from vivarium.types import LookupTableData, NumberLike


def get_calibration_constant_pipeline_name(target_pipeline_name: str) -> str:
    """Return the calibration constant pipeline name for a target pipeline.

    Parameters
    ----------
    target_pipeline_name
        The name of the target pipeline.

    Returns
    -------
        The calibration constant pipeline name in the form
        ``"{target_pipeline_name}.calibration_constant"``.
    """
    return f"{target_pipeline_name}.calibration_constant"


def register_risk_affected_attribute_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    required_resources: Sequence[str] = (),
) -> None:
    """Register a pipeline that can be modified by RiskEffect components.

    Parameters
    ----------
    builder
        Access point for utilizing framework interfaces during setup.
    name
        The name of the pipeline to register.
    source
        The source for the dynamic rate pipeline. This can be a callable
        or a list of column names. If a list of column names is provided,
        the component that is registering this attribute producer must be the
        one that creates those columns.
    required_resources
        A list of resources that the producer requires. A string represents
        a population attribute.
    """
    _RiskAffectedPipeline.create(builder, name, source, required_resources, is_rate=False)


def register_risk_affected_rate_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    required_resources: Sequence[str] = (),
) -> None:
    """Register a pipeline that can be modified by RiskEffect components.

    Parameters
    ----------
    builder
        Access point for utilizing framework interfaces during setup.
    name
        The name of the pipeline to register.
    source
        The source for the dynamic rate pipeline. This can be a callable
        or a list of column names. If a list of column names is provided,
        the component that is registering this attribute producer must be the
        one that creates those columns.
    required_resources
        A list of resources that the producer requires. A string represents
        a population attribute.
    """
    _RiskAffectedPipeline.create(builder, name, source, required_resources, is_rate=True)


class _RiskAffectedPipeline(Component):
    """Convenience class to package pipelines that can be targeted by RiskEffect."""

    @classmethod
    def create(
        cls,
        builder: Builder,
        name: str,
        source: Callable[..., pd.Series],
        required_resources: Sequence[str],
        is_rate: bool,
    ) -> None:
        """Instantiate and set up a ``_RiskAffectedPipeline``."""
        cls(name, source, required_resources, is_rate).setup_component(builder)

    def __init__(
        self,
        target_pipeline_name: str,
        target_pipeline_source: Callable[..., pd.Series],
        required_resources: Sequence[str | Resource],
        is_rate: bool,
    ):
        """Define attributes needed for ``_RiskAffectedPipeline``."""
        super().__init__()
        self._target_pipeline_name = target_pipeline_name
        self._target_pipeline_source = target_pipeline_source
        self._required_resources = required_resources
        self._is_rate = is_rate

    def setup(self, builder: Builder) -> None:
        """Register the calibration constant and target pipelines."""
        self._calibration_constant_table = self.build_lookup_table(
            builder, "calibration_constant", data_source=0
        )
        self._calibration_constant_pipeline = builder.value.register_value_producer(
            get_calibration_constant_pipeline_name(self._target_pipeline_name),
            source=lambda: [0],
            preferred_combiner=self._calibration_constant_combiner,
            preferred_post_processor=self._calibration_constant_post_processor,
        )

        register_pipeline = (
            builder.value.register_rate_producer
            if self._is_rate
            else builder.value.register_attribute_producer
        )

        register_pipeline(
            self._target_pipeline_name,
            source=self._target_pipeline_source,
            required_resources=[self._calibration_constant_table, *self._required_resources],
            preferred_combiner=multiplication_combiner,
            preferred_post_processor=self._apply_calibration_constant,
        )

    def on_post_setup(self, event: Event) -> None:
        """Precompute the calibration constant and store it in the lookup table."""
        calibration_constant_data = self._calibration_constant_pipeline()
        self._calibration_constant_table.set_data(calibration_constant_data)

    @property
    def name(self) -> str:
        """The name of this component."""
        return f"_risk_affected_pipeline.{self._target_pipeline_name}"

    #################################
    # Combiners and post-processors #
    #################################

    @staticmethod
    def _calibration_constant_combiner(
        value: list[Numeric | pd.DataFrame],
        mutator: Callable[..., Numeric | pd.DataFrame],
        *args: Any,
        **kwargs: Any,
    ) -> list[Numeric | pd.Series]:
        """Append the mutator result to the calibration constant list."""
        calibration_constant = mutator(*args, **kwargs)
        if isinstance(calibration_constant, pd.DataFrame):
            index_columns = [
                col for col in calibration_constant.columns if col != DEFAULT_VALUE_COLUMN
            ]
            calibration_constant = calibration_constant.set_index(index_columns).squeeze()
        value.append(calibration_constant)
        return value

    @staticmethod
    def _calibration_constant_post_processor(
        value: list[NumberLike], manager: ValuesManager
    ) -> LookupTableData:
        """Compute the joint calibration constant via raw union."""
        joint_calibration_constant = raw_union_post_processor(value, manager)
        if isinstance(joint_calibration_constant, pd.Series):
            joint_calibration_constant = joint_calibration_constant.reset_index()
        return joint_calibration_constant

    def _apply_calibration_constant(
        self,
        index: pd.Index,
        value: pd.Series,
        manager: ValuesManager,
    ) -> pd.Series:
        """Multiply non-zero values by ``(1 - calibration_constant)``."""
        non_zero_index = value[value != 0].index
        if not non_zero_index.empty:
            calibration_constant = self._calibration_constant_table(non_zero_index)
            value.loc[non_zero_index] = value.loc[non_zero_index] * (1 - calibration_constant)
        return value
