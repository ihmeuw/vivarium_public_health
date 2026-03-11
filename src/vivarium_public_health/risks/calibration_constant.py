"""
====================
Calibration Constant
====================

This module contains functions and classes for managing calibration constants in
pipelines that are intended to be modifiable by RiskEffect components.

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
from vivarium.framework.values import ValuesManager, raw_union_post_processor
from vivarium.types import LookupTableData, NumberLike


def get_calibration_constant_pipeline_name(target_pipeline_name: str) -> str:
    return f"{target_pipeline_name}.calibration_constant"


def register_risk_affected_attribute_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    required_resources: Sequence[str] = (),
) -> None:
    """Helper function to register a pipeline that can be modified by RiskEffect components.

    Parameters
    ----------
    builder
        The Builder object to use for registration.
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
    """Helper function to register a pipeline that can be modified by RiskEffect components.

    Parameters
    ----------
    builder
        The Builder object to use for registration.
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
        """Factory method to create and set up the class."""
        cls(name, source, required_resources, is_rate).setup_component(builder)

    def __init__(
        self,
        target_pipeline_name: str,
        target_pipeline_source: Callable[..., pd.Series],
        required_resources: Sequence[str | Resource],
        is_rate: bool,
    ):
        super().__init__()
        self._target_pipeline_name = target_pipeline_name
        self._target_pipeline_source = target_pipeline_source
        self._required_resources = required_resources
        self._is_rate = is_rate

    def setup(self, builder: Builder) -> None:
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
            preferred_post_processor=self._apply_calibration_constant,
        )

    def on_post_setup(self, event: Event) -> None:
        calibration_constant_data = self._calibration_constant_pipeline()
        self._calibration_constant_table.set_data(calibration_constant_data)

    @property
    def name(self) -> str:
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
        non_zero_index = value[value != 0].index
        if not non_zero_index.empty:
            calibration_constant = self._calibration_constant_table(non_zero_index)
            value.loc[non_zero_index] = value.loc[non_zero_index] * (1 - calibration_constant)
        return value
