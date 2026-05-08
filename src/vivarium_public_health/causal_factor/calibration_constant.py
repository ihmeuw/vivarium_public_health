"""
====================
Calibration Constant
====================

This module contains functions and classes for managing calibration constants in
pipelines that are intended to be modifiable by CausalFactorEffect components. Population
attributable fractions (PAFs) can often be used interchangeably with multiplicative
calibration constants.

"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload
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


@overload
def register_risk_affected_attribute_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
    columns: None = None,
) -> None:
    ...

@overload
def register_risk_affected_attribute_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.DataFrame],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
    columns: list[str] = ...,
) -> None:
    ...

def register_risk_affected_attribute_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series] | Callable[..., pd.DataFrame],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
    columns: list[str] | None = None,
) -> None:
    """Register a pair of pipelines for an attribute targetable by CausalFactorEffect components.

    Two value pipelines are registered:

    * The **target attribute pipeline** (``name``): produces the affected
      attribute. Other components register their effects as modifiers on
      this pipeline.
    * The **calibration constant pipeline**
      (``<name>.calibration_constant``): a companion pipeline that any
      component modifying the target should also modify, registering the
      calibration constant that corresponds to its effect. The joint
      calibration constant is precomputed at the end of setup and
      registered as a modifier on the target pipeline so that the
      population-level baseline is preserved.

    The joint calibration constant is shaped to cancel cleanly against the
    target pipeline's combiner:

    * For ``"multiplicative"`` effects, the joint value is
      ``1 - raw_union(c_i) = prod(1 - c_i)``. Multiplying the target by
      this value scales it by the surviving fraction.
    * For ``"additive"`` effects, the joint value is ``-sum(c_i)``. Adding
      this to the target subtracts the cumulative additive effect.

    Parameters
    ----------
    builder
        The Builder object to use for registration.
    name
        The name of the target pipeline to register. The calibration constant
        pipeline is registered at ``<name>.calibration_constant``.
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
    columns
        If the pipeline should produce a DataFrame, the columns that DataFrame should include.
        None if the pipeline produces a Series.
    """
    _RiskAffectedPipeline.create(
        builder, name, source, effect_type, required_resources, post_processors, columns, is_rate=False
    )


@overload
def register_risk_affected_rate_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
    columns: None = None,
) -> None:
    ...

@overload
def register_risk_affected_rate_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.DataFrame],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
    columns: list[str] = ...,
) -> None:
    ...
def register_risk_affected_rate_producer(
    builder: Builder,
    name: str,
    source: Callable[..., pd.Series],
    effect_type: Literal["multiplicative", "additive"] = "multiplicative",
    required_resources: Sequence[str | Resource] = (),
    post_processors: AttributePostProcessor | Sequence[AttributePostProcessor] = (),
    columns: list[str] | None = None,
) -> None:
    """Register a pair of pipelines for a rate targetable by CausalFactorEffect components.

    Two value pipelines are registered:

    * The **target rate pipeline** (``name``): produces the affected rate.
      Other components register their effects as modifiers on this
      pipeline.
    * The **calibration constant pipeline**
      (``<name>.calibration_constant``): a companion pipeline that any
      component modifying the target should also modify, registering the
      calibration constant that corresponds to its effect. The joint
      calibration constant is precomputed at the end of setup and
      registered as a modifier on the target pipeline so that the
      population-level baseline is preserved.

    The joint calibration constant is shaped to cancel cleanly against the
    target pipeline's combiner:

    * For ``"multiplicative"`` effects, the joint value is
      ``1 - raw_union(c_i) = prod(1 - c_i)``. Multiplying the target by
      this value scales it by the surviving fraction.
    * For ``"additive"`` effects, the joint value is ``-sum(c_i)``. Adding
      this to the target subtracts the cumulative additive effect.

    Parameters
    ----------
    builder
        The Builder object to use for registration.
    name
        The name of the target pipeline to register. The calibration constant
        pipeline is registered at ``<name>.calibration_constant``.
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
    columns
        If the pipeline should produce a DataFrame, the columns that DataFrame should include.
        None if the pipeline produces a Series.
    """
    _RiskAffectedPipeline.create(
        builder, name, source, effect_type, required_resources, post_processors, columns, is_rate=True
    )


class _RiskAffectedPipeline(Component):
    """Package a pair of pipelines that can be targeted by CausalFactorEffect components.

    Two value pipelines are registered during ``setup``:

    * The **target pipeline** (``<name>``): produces the affected attribute or
      rate. Other components register their effects as modifiers on this
      pipeline. The combiner used depends on ``effect_type`` —
      :func:`~vivarium.framework.values.multiplication_combiner` for
      ``"multiplicative"`` effects and
      :func:`~vivarium.framework.values.addition_combiner` for ``"additive"``
      effects.
    * The **calibration constant pipeline**
      (``<name>.calibration_constant``): a companion pipeline that any
      component modifying the target should also modify, registering the
      calibration constant that corresponds to its effect. The collected
      constants ``c_i`` are reduced to a single joint value (see below).
      In ``on_post_setup`` the joint value is materialized into
      :attr:`_calibration_constant_table` and registered as a modifier on
      the target pipeline.

    The joint calibration constant is shaped to cancel cleanly against the
    target pipeline's combiner, preserving the population-level baseline:

    * For ``"multiplicative"`` effects, the post-processor returns
      ``1 - raw_union(c_i) = prod(1 - c_i)``. Multiplying the target by this
      value scales it by the surviving fraction.
    * For ``"additive"`` effects, the post-processor returns ``-sum(c_i)``.
      Adding this to the target subtracts the cumulative additive effect.
    """

    @classmethod
    def create(
        cls,
        builder: Builder,
        name: str,
        source: Callable[..., pd.Series],
        effect_type: Literal["multiplicative", "additive"],
        required_resources: Sequence[str | Resource],
        post_processors: AttributePostProcessor | Sequence[AttributePostProcessor],
        columns: list[str] | None,
        is_rate: bool,
    ) -> _RiskAffectedPipeline:
        """Factory method to create and set up the class."""
        pipeline = cls(
            name, source, effect_type, required_resources, post_processors, columns, is_rate
        )
        pipeline.setup_component(builder)
        return pipeline

    def __init__(
        self,
        target_pipeline_name: str,
        target_pipeline_source: Callable[..., pd.Series],
        effect_type: Literal["multiplicative", "additive"],
        required_resources: Sequence[str | Resource],
        post_processors: AttributePostProcessor | Sequence[AttributePostProcessor],
        columns: list[str] | None,
        is_rate: bool,
    ):
        super().__init__()
        self._target_pipeline_name = target_pipeline_name
        """Name of the target pipeline."""
        self._target_pipeline_source = target_pipeline_source
        """Callable that produces values for the target pipeline."""
        self._effect_type = effect_type
        """``"multiplicative"`` or ``"additive"``. Determines the combiner used on the
        target pipeline and the reduction used to compute the joint calibration
        constant."""
        self._required_resources = required_resources
        """Resources the target pipeline depends on."""
        self._post_processors = (
            post_processors if isinstance(post_processors, Sequence) else [post_processors]
        )
        """AttributePostProcessors applied to the target pipeline after the combiner
        runs. Normalized to a sequence even when a single post-processor is supplied."""
        self._columns = columns if columns is not None else DEFAULT_VALUE_COLUMN
        """If the pipeline should produce a DataFrame, the columns that DataFrame should include.
        ``DEFAULT_VALUE_COLUMN`` if the pipeline produces a Series."""
        self._is_rate = is_rate
        """``True`` if the target pipeline is a rate, ``False`` if it is an attribute.
        Selects between ``register_rate_producer`` and ``register_attribute_producer``."""
    
    def setup(self, builder: Builder) -> None:
        """Register the calibration constant and target pipelines.

        Parameters
        ----------
        builder
            Access point for utilizing framework interfaces during setup.
        """
        initial_data: int | list[int] = (
            0 if isinstance(self._columns, str) else [0] * len(self._columns)
        )
        self._calibration_constant_table = self.build_lookup_table(
            builder,
            "calibration_constant",
            data_source=initial_data,
            value_columns=self._columns,
        )
        """Lookup table populated in ``on_post_setup`` with the precomputed joint
        calibration constant. Registered as a modifier on the target pipeline so the
        constant is applied uniformly without re-running the reduction each time-step."""
        self._calibration_constant_pipeline = builder.value.register_value_producer(
            get_calibration_constant_pipeline_name(self._target_pipeline_name),
            source=lambda: [0],
            preferred_combiner=self._calibration_constant_combiner,
            preferred_post_processor=self._calibration_constant_post_processor,
        )
        """Value pipeline that produces the joint calibration constant by reducing
        the list of registered calibration constants."""

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

        # Register the calibration constant table as a modifier on the target pipeline.
        # Calibrations constant application is commutative with other modifiers, so the
        # order of application does not matter.
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

    def _calibration_constant_combiner(
        self,
        value: list[Numeric | pd.DataFrame],
        mutator: Callable[..., Numeric | pd.DataFrame],
        *args: Any,
        **kwargs: Any,
    ) -> list[Numeric | pd.Series]:
        """Append the mutator result to the calibration constant list."""
        calibration_constant = mutator(*args, **kwargs)
        if isinstance(calibration_constant, pd.DataFrame):
            value_columns = self._columns if not isinstance(self._columns, str) else [self._columns]
            index_columns = [
                col for col in calibration_constant.columns if col not in value_columns
            ]
            calibration_constant = calibration_constant.set_index(index_columns).squeeze()
        value.append(calibration_constant)
        return value

    def _calibration_constant_post_processor(
        self, value: list[NumberLike], manager: ValuesManager
    ) -> LookupTableData:
        """Reduce the list of registered calibration constants to a single joint value.

        The returned value is shaped so that it cancels the cumulative effects
        when applied to the target pipeline by the corresponding combiner,
        preserving the population-level baseline.

        * ``"multiplicative"`` effects: returns ``1 - raw_union(c_i)``, which
          equals ``prod(1 - c_i)``. Applied via
          :func:`~vivarium.framework.values.multiplication_combiner`, this
          scales the target by the surviving fraction.
        * ``"additive"`` effects: returns ``-sum(c_i)``. The negation is
          essential — applied via
          :func:`~vivarium.framework.values.addition_combiner`, this
          subtracts the cumulative additive effect from the target.

        See the class docstring for the broader context.
        """
        if self._effect_type == "multiplicative":
            joint_calibration_constant = 1 - raw_union_post_processor(value, manager)
        elif self._effect_type == "additive":
            joint_calibration_constant = -sum(value)
        else:
            raise ValueError(f"Unsupported effect type: {self._effect_type}")
        if isinstance(joint_calibration_constant, pd.Series):
            joint_calibration_constant = joint_calibration_constant.reset_index()
        elif isinstance(joint_calibration_constant, pd.DataFrame):
            if isinstance(joint_calibration_constant.index, pd.MultiIndex):
                joint_calibration_constant = joint_calibration_constant.reset_index()
        elif not isinstance(self._columns, str):
            # Scalar joint with a multi-column pipeline —
            # broadcast across columns so the lookup table can ingest one value per column.
            joint_calibration_constant = [joint_calibration_constant] * len(self._columns)
        return joint_calibration_constant
