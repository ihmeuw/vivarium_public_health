"""
=========
Utilities
=========

This module contains utility classes and functions for use across
vivarium_public_health components.

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


def get_joint_paf_pipeline_name(target_pipeline_name: str) -> str:
    return f"{target_pipeline_name}.paf"


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
    pipeline_component = _RiskAffectedPipeline(
        name, source, required_resources, is_rate=False
    )
    pipeline_component.setup_component(builder)


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
    pipeline_component = _RiskAffectedPipeline(name, source, required_resources, is_rate=True)
    pipeline_component.setup_component(builder)


class _RiskAffectedPipeline(Component):
    """Convenience class to package pipelines that can be targeted by RiskEffect."""

    def __init__(
        self,
        target_pipeline_name: str,
        target_pipeline_source: Callable[..., pd.Series],
        required_resources: Sequence[str | Resource],
        is_rate: bool,
    ):
        super().__init__()
        self.target_pipeline_name = target_pipeline_name
        self.target_pipeline_source = target_pipeline_source
        self.required_resources = required_resources
        self.is_rate = is_rate

    def setup(self, builder: Builder) -> None:
        self.joint_paf_table = self.build_lookup_table(builder, "joint_paf", data_source=0)
        self.joint_paf_pipeline = builder.value.register_value_producer(
            get_joint_paf_pipeline_name(self.target_pipeline_name),
            source=lambda: [0],
            preferred_combiner=self._joint_paf_combiner,
            preferred_post_processor=self._joint_paf_post_processor,
        )

        register_pipeline = (
            builder.value.register_rate_producer
            if self.is_rate
            else builder.value.register_attribute_producer
        )

        register_pipeline(
            self.target_pipeline_name,
            source=self.target_pipeline_source,
            required_resources=[self.joint_paf_table, *self.required_resources],
            preferred_post_processor=self._apply_joint_paf,
        )

    def on_post_setup(self, event: Event) -> None:
        paf_data = self.joint_paf_pipeline()
        self.joint_paf_table.set_data(paf_data)

    #################################
    # Combiners and post-processors #
    #################################

    @staticmethod
    def _joint_paf_combiner(
        value: list[Numeric | pd.DataFrame],
        mutator: Callable[..., Numeric | pd.DataFrame],
        *args: Any,
        **kwargs: Any,
    ) -> list[Numeric | pd.Series]:
        paf = mutator(*args, **kwargs)
        if isinstance(paf, pd.DataFrame):
            index_columns = [col for col in paf.columns if col != DEFAULT_VALUE_COLUMN]
            paf = paf.set_index(index_columns).squeeze()
        value.append(paf)
        return value

    @staticmethod
    def _joint_paf_post_processor(
        value: list[NumberLike], manager: ValuesManager
    ) -> LookupTableData:
        joint_paf = raw_union_post_processor(value, manager)
        if isinstance(joint_paf, pd.Series):
            joint_paf = joint_paf.reset_index()
        return joint_paf

    def _apply_joint_paf(
        self,
        index: pd.Index,
        value: pd.Series,
        manager: ValuesManager,
    ) -> pd.Series:
        non_zero_index = value[value != 0].index
        if not non_zero_index.empty:
            joint_paf = self.joint_paf_table(non_zero_index)
            value.loc[non_zero_index] = value.loc[non_zero_index] * (1 - joint_paf)
        return value
