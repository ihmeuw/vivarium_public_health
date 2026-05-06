"""
====================
Calibration Constant
====================

This module contains functions and classes for managing calibration constants in
pipelines that are intended to be modifiable by RiskEffect components. Population
attributable fractions (PAFs) can often be used interchangeably with calibration
constants.

"""
from collections.abc import Callable, Sequence
from typing import Any, Literal
from typing import SupportsFloat as Numeric

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import DEFAULT_VALUE_COLUMN
from vivarium.framework.resource import Resource
from vivarium.framework.values import (
    AttributePostProcessor,
    ValuesManager,
    addition_combiner,
    multiplication_combiner,
    raw_union_post_processor,
)
from vivarium.types import LookupTableData, NumberLike


def get_calibration_constant_pipeline_name(target_pipeline_name: str) -> str:
    """Return the calibration constant pipeline name for a target pipeline."""
    return f"{target_pipeline_name}.calibration_constant"


def register_risk_affected_attribute_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
) -> None:
    """Helper function to register a pipeline that can be modified by CausalFactorEffect components.

    Parameters
    ----------
    builder
        The Builder object to use for registration.
    name
        The name of the pipeline to register.
    source
        The source for the attribute pipeline. This can be a callable
        or a list of column names. If a list of column names is provided,
        the component that is registering this attribute producer must be the
        one that creates those columns.
    effect_type
        The type of effect that CausalFactorEffect components will have on this pipeline.
        Supported values are "multiplicative" and "additive". This will determine how
        modifiers are applied to the pipeline in conjunction with the calibration constant.
    required_resources
        A list of resources that the producer requires. A string represents
        a population attribute.
    post_processors
        An AttributePostProcessor or list of AttributePostProcessors to apply
        to the attribute pipeline.
    """
    post_processors = (
        post_processors if isinstance(post_processors, Sequence) else [post_processors]
    )
    _RiskAffectedPipeline.create(
        builder,
        name,
        source,
        effect_type,
        required_resources,
        post_processors,
        is_rate=False,
    )


def register_risk_affected_rate_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
) -> None:
    """Helper function to register a rate pipeline that can be modified by CausalFactorEffect components.

    Parameters
    ----------
    builder
        The Builder object to use for registration.
    name
        The name of the pipeline to register.
    source
        The source for the rate pipeline. This can be a callable
        or a list of column names. If a list of column names is provided,
        the component that is registering this rate producer must be the
        one that creates those columns.
    effect_type
        The type of effect that CausalFactorEffect components will have on this pipeline.
        Supported values are "multiplicative" and "additive". This will determine how
        modifiers are applied to the pipeline in conjunction with the calibration constant.
    required_resources
        A list of resources that the producer requires. A string represents
        a population attribute.
    post_processors
        An AttributePostProcessor or list of AttributePostProcessors to apply
        to the attribute pipeline.
    """
    post_processors = (
        post_processors if isinstance(post_processors, Sequence) else [post_processors]
    )
    _RiskAffectedPipeline.create(
        builder, name, source, effect_type, required_resources, post_processors, is_rate=True
    )


class _RiskAffectedPipeline(Component):
    """Convenience class to package pipelines that can be targeted by CausalFactorEffect."""

    @classmethod
    def create(
        cls,
        builder: Builder,
        name: str,
        source: Callable[..., pd.Series],
        effect_type: Literal["multiplicative", "additive"],
        required_resources: Sequence[str | Resource],
        post_processors: Sequence[AttributePostProcessor],
        is_rate: bool,
    ) -> None:
        """Factory method to create and set up the class."""
        cls(
            name, source, effect_type, required_resources, post_processors, is_rate
        ).setup_component(builder)

    def __init__(
        self,
        target_pipeline_name: str,
        target_pipeline_source: Callable[..., pd.Series],
        effect_type: Literal["multiplicative", "additive"],
        required_resources: Sequence[str | Resource],
        post_processors: Sequence[AttributePostProcessor],
        is_rate: bool,
    ):
        super().__init__()
        self._target_pipeline_name = target_pipeline_name
        self._target_pipeline_source = target_pipeline_source
        self._effect_type = effect_type
        self._required_resources = required_resources
        self._post_processors = post_processors
        self._is_rate = is_rate

    def setup(self, builder: Builder) -> None:
        """Register the calibration constant and target pipelines.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
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
        combiner = (
            multiplication_combiner
            if self._effect_type == "multiplicative"
            else addition_combiner
        )

        register_pipeline(
            self._target_pipeline_name,
            source=self._target_pipeline_source,
            required_resources=self._required_resources,
            preferred_combiner=combiner,
            preferred_post_processor=self._post_processors,
        )

        builder.value.register_attribute_modifier(
            self._target_pipeline_name, modifier=self._calibration_constant_table
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

    def _calibration_constant_post_processor(
        self, value: list[NumberLike], manager: ValuesManager
    ) -> LookupTableData:
        """Compute the joint calibration constant.

        Uses a union post processor if the effect type is multiplicative and an addition
        post processor if the effect type is additive.


        """
        if self._effect_type == "multiplicative":
            joint_calibration_constant = 1 - raw_union_post_processor(value, manager)
        elif self._effect_type == "additive":
            joint_calibration_constant = -sum(value)
        else:
            raise ValueError(f"Unsupported effect type: {self._effect_type}")
        if isinstance(joint_calibration_constant, pd.Series):
            joint_calibration_constant = joint_calibration_constant.reset_index()
        return joint_calibration_constant
